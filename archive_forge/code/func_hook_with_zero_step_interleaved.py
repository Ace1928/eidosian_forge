import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def hook_with_zero_step_interleaved(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool=False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`
    optimizer step with the :class:`DistributedDataParallel` backward pass,
    where the optimizer step computation interleaves with the backward
    computation.

    This approach overlaps the optimizer computation and communication with the
    backward computation and communication. In particular, once a bucket's
    gradients have been computed, the optimizer computation using those
    gradients is launched (though the actual computation must wait for the
    bucket's all-reduce to complete). This yields an interleaving of all-
    reduces and broadcasts in the communication stream.

    This approach may be preferred over :meth:`hook_with_zero_step` if
    communication is relatively fast compared to computation.

    Arguments:
        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
            modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError('ZeroRedundancyOptimizer must be constructed with `overlap_with_ddp=True` to use this hook properly')
    ddp_ref = weakref.ref(ddp)
    pg = dist.get_backend(ddp_ref().process_group)
    if pg != dist.Backend.NCCL and pg != 'hccl':
        raise RuntimeError('Overlapping DDP with ZeRO using this approach currently requires NCCL/HCCL backend to avoid hangs')
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_interleaved_fn(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        """
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step` using
        the gradients in that bucket.
        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            """
            Performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step`
            using the gradients in the given :class:`DistributedDataParallel`
            gradient bucket.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)
            _broadcast_bucket(bucket_index, zero)
            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()
            return bucket.buffer()
        return fut.then(zero_step)
    return hook_with_zero_interleaved_fn