import logging
import threading
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple
import ray
from ray.actor import ActorHandle
from ray.data._internal.batcher import Batcher, ShufflingBatcher
from ray.data._internal.block_batching.interfaces import (
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def finalize_batches(batch_iter: Iterator[CollatedBatch], finalize_fn: Callable[[Any], Any], stats: Optional[DatasetStats]=None) -> Iterator[CollatedBatch]:
    """Returns an iterator with the provided finalize_fn applied to items of the batch
    iterator.

    This is the same as `collate` except the input batches can be of type Any.

    Args:
        batch_iter: An iterator over processed batches.
        finalize_fn: A function to apply to each batch.
        stats: An optional stats object to record formatting times.

    Returns:
        An iterator over batch index and the finalized batch.
    """
    for batch in batch_iter:
        with stats.iter_finalize_batch_s.timer() if stats else nullcontext():
            finalized_batch = finalize_fn(batch.data)
        yield CollatedBatch(batch.batch_idx, finalized_batch)