import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def hook_with_zero_fn(state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs the equivalent of a :class:`ZeroRedundancyOptimizer`
        :meth:`step` if ``bucket`` is the last gradient bucket.

        The function performs additional computation on the iteration that
        the :class:`DistributedDataParallel` buckets are rebuilt to collect
        information used to implement the modified hook.

        Arguments:
            state (Any): any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
    fut = hook(state, bucket)
    _hook_with_zero_step_setup(ddp_ref, zero, bucket)
    if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
        return fut
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    rank = zero.global_rank
    assert overlap_info.status == _OverlapStatus.INITIALIZED
    assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, '`assigned_ranks_per_bucket` is not fully constructed'
    assigned_to_bucket = rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
    if assigned_to_bucket:
        overlap_info.bucket_index_to_bucket[bucket_index] = bucket
        overlap_info.bucket_index_to_future[bucket_index] = fut
    if len(overlap_info.bucket_indices_seen) > 0:
        assert overlap_info.bucket_indices_seen[-1] == bucket_index - 1, 'Bucket indices are not in incremental order'
    else:
        assert bucket_index == 0, 'Bucket indices do not start from 0'
    overlap_info.bucket_indices_seen.append(bucket_index)
    num_buckets = len(overlap_info.params_per_bucket)
    is_last_bucket = bucket_index == num_buckets - 1
    if not is_last_bucket:
        return fut
    for bucket_index in range(num_buckets):
        assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
        if rank in assigned_ranks:
            assert bucket_index in overlap_info.bucket_index_to_future, f'All-reduce future for bucket {bucket_index} not saved on rank {rank}'
            allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
            allreduce_future.wait()
            curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
            _perform_local_step(curr_bucket, zero, rank)
        _broadcast_bucket(bucket_index, zero)
    overlap_info.wait_for_broadcasts()
    overlap_info.clear_per_iter_info()
    return fut