from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:

        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.

        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).

    2. Handles uncompressed tensors:

        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;

        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.

    3. Handles the tensors that should be compressed by PowerSGD compression:

        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

        3.2. Computes each P in Ps, which is equal to MQ;

        3.3. Allreduces Ps as a batch;

        3.4. Orthogonalizes each P in Ps;

        3.5. Computes each Q in Qs, which is approximately equal to M^TP;

        3.6. Allreduces Qs as a batch;

        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``
            and ``min_compression_rate``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,
                                  start_powerSGD_iter=10, min_compression_rate=0.5)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    input_tensor = bucket.buffer()
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)
    device = input_tensor.device
    dtype = input_tensor.dtype
    bucket_index = bucket.index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info('A zero tensor of length %s that represents local error is created.', total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
        input_tensor_cp = torch.clone(input_tensor).detach()
    tensors = bucket.gradients()
    tensors_to_compress, uncompressed_tensors = ([], [])
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(n, m, matrix_approximation_rank, state.min_compression_rate)
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]
    _report_compression_stats(bucket, state)
    uncompressed_tensors_memory = torch.cat([tensor.view(-1) for tensor in uncompressed_tensors]) if uncompressed_tensors else torch.tensor([], device=device, dtype=dtype)
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        if state.warm_start:
            logger.info('Allocating contiguous memory of length %s for Ps, and of length %s for Qs, respectively.', total_Ps_size, total_Qs_size)
        state.p_memory_dict[bucket_index] = torch.empty(total_Ps_size, device=device, dtype=dtype)
        state.q_memory_dict[bucket_index] = torch.empty(total_Qs_size, device=device, dtype=dtype)
    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        shape_to_tensors[tensor.shape].append(tensor)

    def maybe_batched_tensors_to_compress():
        for tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                batch_size = len(tensors)
                if batch_size == 1:
                    yield tensors[0].unsqueeze(0)
                else:
                    yield torch.stack(tensors)
            else:
                for tensor in tensors:
                    yield tensor.unsqueeze(0)
    tensors_to_compress = []
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in maybe_batched_tensors_to_compress():
        batch_size, n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        tensors_to_compress.append(tensor)
        ps.append(state.p_memory_dict[bucket_index][p_idx:p_idx + batch_size * n * matrix_approximation_rank].view(batch_size, n, matrix_approximation_rank))
        qs.append(state.q_memory_dict[bucket_index][q_idx:q_idx + batch_size * m * matrix_approximation_rank].view(batch_size, m, matrix_approximation_rank))
        p_idx += batch_size * n * matrix_approximation_rank
        q_idx += batch_size * m * matrix_approximation_rank
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state.rng.randint(1000000000))
            for q in qs:
                q.copy_(torch.randn(*q.shape, device='cpu', dtype=dtype))
                _orthogonalize(q, state.orthogonalization_epsilon)
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
        torch.bmm(tensor, q, out=p)
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(uncompressed_tensors_memory, group=group_to_use, async_op=True).get_future()

    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(uncompressed_tensors_memory[idx:idx + tensor.numel()].view_as(tensor))
            idx += tensor.numel()
        return dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)
        for tensor, p, q in zip(tensors_to_compress, ps, qs):
            torch.bmm(tensor.transpose(1, 2), p, out=q)
        return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        for p, q, tensor in zip(ps, qs, tensors_to_compress):
            torch.bmm(p, q.transpose(1, 2), out=tensor)
        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if state.use_error_feedback:
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        state.maybe_increase_iter(bucket)
        return input_tensor
    return allreduce_contiguous_uncompressed_tensors_fut.then(unpack_uncompressed_tensors_and_allreduce_ps).then(compute_qs).then(decompress)