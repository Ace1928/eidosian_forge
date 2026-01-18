from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
def next_batch(self) -> Block:
    """Get the next shuffled batch from the shuffle buffer.

        Returns:
            A batch represented as a Block.
        """
    assert self.has_batch() or (self._done_adding and self.has_any())
    if self._builder.num_rows() > 0 and (self._done_adding or self._materialized_buffer_size() <= self._buffer_min_size):
        if self._shuffle_buffer is not None:
            if self._batch_head > 0:
                block = BlockAccessor.for_block(self._shuffle_buffer)
                self._shuffle_buffer = block.slice(self._batch_head, block.num_rows())
            self._builder.add_block(self._shuffle_buffer)
        self._shuffle_buffer = self._builder.build()
        self._shuffle_buffer = BlockAccessor.for_block(self._shuffle_buffer).random_shuffle(self._shuffle_seed)
        if self._shuffle_seed is not None:
            self._shuffle_seed += 1
        if isinstance(BlockAccessor.for_block(self._shuffle_buffer), ArrowBlockAccessor) and self._shuffle_buffer.num_columns > 0 and (self._shuffle_buffer.column(0).num_chunks >= MIN_NUM_CHUNKS_TO_TRIGGER_COMBINE_CHUNKS):
            self._shuffle_buffer = transform_pyarrow.combine_chunks(self._shuffle_buffer)
        self._builder = DelegatingBlockBuilder()
        self._batch_head = 0
    assert self._shuffle_buffer is not None
    buffer_size = BlockAccessor.for_block(self._shuffle_buffer).num_rows()
    batch_size = min(self._batch_size, buffer_size)
    slice_start = self._batch_head
    self._batch_head += batch_size
    return BlockAccessor.for_block(self._shuffle_buffer).slice(slice_start, self._batch_head)