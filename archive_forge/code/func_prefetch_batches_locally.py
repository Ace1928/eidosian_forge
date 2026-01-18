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
def prefetch_batches_locally(block_ref_iter: Iterator[Tuple[ObjectRef[Block], BlockMetadata]], prefetcher: BlockPrefetcher, num_batches_to_prefetch: int, batch_size: Optional[int], eager_free: bool=False) -> Iterator[ObjectRef[Block]]:
    """Given an iterator of batched block references, returns an iterator over the same
    block references while prefetching `num_batches_to_prefetch` batches in advance.

    Args:
        block_ref_iter: An iterator over batched block references.
        prefetcher: The prefetcher to use.
        num_batches_to_prefetch: The number of batches to prefetch ahead of the
            current batch during the scan.
        batch_size: User specified batch size, or None to let the system pick.
        eager_free: Whether to eagerly free the object reference from the object store.
    """
    sliding_window = collections.deque()
    current_window_size = 0
    if num_batches_to_prefetch <= 0:
        for block_ref, metadata in block_ref_iter:
            yield block_ref
        return
    if batch_size is not None:
        num_rows_to_prefetch = num_batches_to_prefetch * batch_size
    else:
        num_rows_to_prefetch = None
    while batch_size is not None and current_window_size < num_rows_to_prefetch or (batch_size is None and len(sliding_window) < num_batches_to_prefetch):
        try:
            next_block_ref_and_metadata = next(block_ref_iter)
        except StopIteration:
            break
        sliding_window.append(next_block_ref_and_metadata)
        current_window_size += next_block_ref_and_metadata[1].num_rows
    prefetcher.prefetch_blocks([block_ref for block_ref, _ in list(sliding_window)])
    while sliding_window:
        block_ref, metadata = sliding_window.popleft()
        current_window_size -= metadata.num_rows
        if batch_size is None or current_window_size < num_rows_to_prefetch:
            try:
                sliding_window.append(next(block_ref_iter))
                prefetcher.prefetch_blocks([block_ref for block_ref, _ in list(sliding_window)])
            except StopIteration:
                pass
        yield block_ref
        trace_deallocation(block_ref, loc='iter_batches', free=eager_free)
    prefetcher.stop()