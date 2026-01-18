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
def collate(batch_iter: Iterator[Batch], collate_fn: Optional[Callable[[DataBatch], Any]], stats: Optional[DatasetStats]=None) -> Iterator[CollatedBatch]:
    """Returns an iterator with the provided collate_fn applied to items of the batch
    iterator.

    Args:
        batch_iter: An iterator over formatted batches.
        collate_fn: A function to apply to each batch.
        stats: An optional stats object to record formatting times.
    """
    for batch in batch_iter:
        with stats.iter_collate_batch_s.timer() if stats else nullcontext():
            collated_batch = collate_fn(batch.data)
        yield CollatedBatch(batch.batch_idx, collated_batch)