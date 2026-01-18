import collections
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import ray
from ray.data._internal.block_batching.interfaces import Batch, BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import make_async_gen
from ray.data.block import Block, BlockMetadata, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def iter_batches(block_refs: Iterator[Tuple[ObjectRef[Block], BlockMetadata]], *, stats: Optional[DatasetStats]=None, clear_block_after_read: bool=False, batch_size: Optional[int]=None, batch_format: Optional[str]='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], Any]]=None, finalize_fn: Optional[Callable[[Any], Any]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False, prefetch_batches: int=1) -> Iterator[DataBatch]:
    """Create formatted batches of data from an iterator of block object references and
    corresponding metadata.

    This takes a block iterator and creates batch_size batches, slicing,
    unioning, shuffling, prefetching, and formatting blocks as needed.

    The algorithm uses both pipeline parallelism and data parallelism:

    If prefetch_batches=2, these are all the batches in flight:

    [User thread] trains on Batch 0
    - [Fetch thread] Batch 1 finalization + move to output queue
            - [Worker thread 1] Batch 2 formatting + collating
            - [Worker thread 2] Batch 3 formatting + collating
            - [Raylet] Batches 4 + 5 fetched to local object store memory

    At any point in time there are prefetch_batches+1 batches in local heap memory.
    And the next set of prefetch_batches in local object store memory.

    The actual steps are as follows:

    In a single async thread, do the following:
        1. Trigger Ray local prefetching of `prefetch_batches` worth of block object
            references.
        2. Resolve (i.e. call `ray.get()`) on the block references.
        3. Perform the necessary batch slicing to construct full batches, possibly
            shuffling if necessary.
        4. Then, in a threadpool consisting of `prefetch_batches` threads:
            a. Format the batches to the provided batch format.
            b. Apply the collate function.
        5. Finalize each of the collated batches
        6. Fetch outputs from the threadpool, maintaining order of the batches.

    Args:
        block_refs: An iterator over block object references and their corresponding
            metadata.
        stats: DatasetStats object to record timing and other statistics.
        clear_block_after_read: Whether to clear the block from object store
            manually (i.e. without waiting for Python's automatic GC) after it
            is read. Doing so will reclaim memory faster and hence reduce the
            memory footprint. However, the caller has to ensure the safety, i.e.
            the block will never be accessed again.
        batch_size: Record batch size, or None to let the system pick.
        batch_format: The format in which to return each batch.
            Specify "default" to use the current block format (promoting
            Arrow to pandas automatically), "pandas" to
            select ``pandas.DataFrame`` or "pyarrow" to select
            ``pyarrow.Table``, or None to use entire blocks
            as batches. Default is "default".
        drop_last: Whether to drop the last batch if it's incomplete.
        collate_fn: A function to apply to each data batch before returning it.
        finalize_fn: A function to apply to each data batch after it has been collated.
            This function is not run in a threadpool so it can be used for
            memory-intensive operations such as GPU preloading.
        shuffle_buffer_min_size: If non-None, the data will be randomly shuffled using a
            local in-memory shuffle buffer, and this value will serve as the minimum
            number of rows that must be in the local in-memory shuffle buffer in order
            to yield a batch.
        shuffle_seed: The seed to use for the local random shuffle.
        ensure_copy: Whether batches are always copied from the underlying base
            blocks (not zero-copy views).
        prefetch_batches: The number of batches to fetch ahead of the current batch to
            process. If set to greater than 0, a separate thread will be used to fetch
            the specified amount of formatted batches from blocks. This improves
            performance for non-CPU bound UDFs, allowing batch fetching compute and
            formatting to be overlapped with the UDF. Defaults to 1.

    Returns:
        An iterator over record batches.
    """
    context = DataContext.get_current()
    if prefetch_batches > 0 and context.actor_prefetcher_enabled and (not ray.util.client.ray.is_connected()):
        prefetcher = ActorBlockPrefetcher()
    else:
        prefetcher = WaitBlockPrefetcher()
    eager_free = clear_block_after_read and DataContext.get_current().eager_free

    def _async_iter_batches(block_refs: Iterator[Tuple[ObjectRef[Block], BlockMetadata]]) -> Iterator[DataBatch]:
        block_refs = prefetch_batches_locally(block_ref_iter=block_refs, prefetcher=prefetcher, num_batches_to_prefetch=prefetch_batches, batch_size=batch_size, eager_free=eager_free)
        block_iter = resolve_block_refs(block_ref_iter=block_refs, stats=stats)
        batch_iter = blocks_to_batches(block_iter=block_iter, stats=stats, batch_size=batch_size, drop_last=drop_last, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy)
        batch_iter = _format_in_threadpool(batch_iter, stats=stats, batch_format=batch_format, collate_fn=collate_fn, num_threadpool_workers=prefetch_batches)
        if finalize_fn is not None:
            batch_iter = finalize_batches(batch_iter, finalize_fn=finalize_fn, stats=stats)
        batch_iter: Iterator[Batch] = restore_original_order(batch_iter)
        yield from extract_data_from_batch(batch_iter)
    async_batch_iter = make_async_gen(block_refs, fn=_async_iter_batches, num_workers=1)
    while True:
        with stats.iter_total_blocked_s.timer() if stats else nullcontext():
            try:
                next_batch = next(async_batch_iter)
            except StopIteration:
                break
        with stats.iter_user_s.timer() if stats else nullcontext():
            yield next_batch