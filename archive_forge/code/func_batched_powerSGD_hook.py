from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def batched_powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    This variant does not compress the gradients layer by layer,
    but instead compresses the flattened input tensor that batches all the gradients.
    Therefore, it is **faster** than :meth:`powerSGD_hook`,
    but usually results in a **much lower accuracy**, unless ``matrix_approximation_rank`` is 1.

    .. warning ::
        Increasing ``matrix_approximation_rank`` here may not necessarily increase the accuracy,
        because batching per-parameter tensors without column/row alignment can destroy low-rank structure.
        Therefore, the user should always consider :meth:`powerSGD_hook` first,
        and only consider this variant when a satisfactory accuracy can be achieved when ``matrix_approximation_rank`` is 1.

    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;

    2. Creates two low-rank tensors P and Q for decomposing M, such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

    3. Computes P, which is equal to MQ;

    4. Allreduces P;

    5. Orthogonalizes P;

    6. Computes Q, which is approximately equal to M^TP;

    7. Allreduces Q;

    8. Computes M, which is approximately equal to PQ^T.

    9. Truncates the input tensor to the original length.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank`` and ``start_powerSGD_iter``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    input_tensor = bucket.buffer()
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)
    device = input_tensor.device
    total_length = input_tensor.shape[0]
    state.total_numel_before_compression += total_length
    square_side_length = math.ceil(math.sqrt(total_length))
    state.total_numel_after_compression += square_side_length * state.matrix_approximation_rank * 2
    padded_total_length = square_side_length ** 2
    input_tensor.resize_(padded_total_length)
    input_tensor[total_length:padded_total_length].fill_(0)
    _report_compression_stats(bucket, state)
    bucket_index = bucket.index()
    input_tensor_cp = None
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info('A zero tensor of length %s that represents local error is created.', padded_total_length)
            state.error_dict[bucket_index] = torch.zeros(padded_total_length, device=device, dtype=input_tensor.dtype)
        input_tensor_cp = torch.clone(input_tensor).detach()
    matrix = input_tensor.view(square_side_length, square_side_length)
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        if state.warm_start:
            logger.info('Initializing low-rank tensors P and Q, each of which has a shape of %s x %s.', square_side_length, state.matrix_approximation_rank)

        def create_low_rank_tensor(fill_random_values, rng):
            """Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank."""
            if fill_random_values:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(rng.randint(1000000000))
                    return torch.randn(square_side_length, state.matrix_approximation_rank, device='cpu', dtype=input_tensor.dtype).to(device)
            else:
                return torch.empty(square_side_length, state.matrix_approximation_rank, device=device, dtype=input_tensor.dtype)
        state.p_memory_dict[bucket_index] = create_low_rank_tensor(fill_random_values=False, rng=state.rng)
        state.q_memory_dict[bucket_index] = create_low_rank_tensor(fill_random_values=True, rng=state.rng)
    _orthogonalize(state.q_memory_dict[bucket_index])
    torch.matmul(matrix, state.q_memory_dict[bucket_index], out=state.p_memory_dict[bucket_index])
    allreduce_p_fut = dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future()

    def compute_q(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        _orthogonalize(state.p_memory_dict[bucket_index])
        torch.matmul(matrix.t(), state.p_memory_dict[bucket_index], out=state.q_memory_dict[bucket_index])
        return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        torch.matmul(state.p_memory_dict[bucket_index], state.q_memory_dict[bucket_index].t(), out=matrix)
        if state.use_error_feedback:
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        ret = input_tensor.resize_(total_length)
        state.maybe_increase_iter(bucket)
        return ret
    return allreduce_p_fut.then(compute_q).then(decompress)