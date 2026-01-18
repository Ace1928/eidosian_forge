from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
def has_batch(self) -> bool:
    """Whether this batcher has any batches."""
    buffer_size = self._buffer_size()
    if not self._done_adding:
        return self._materialized_buffer_size() >= self._buffer_min_size or buffer_size - self._batch_size >= self._buffer_min_size * SHUFFLE_BUFFER_COMPACTION_RATIO
    else:
        return buffer_size >= self._batch_size