from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
def done_adding(self) -> bool:
    """Indicate to the batcher that no more blocks will be added to the batcher.

        No more blocks should be added to the batcher after calling this.
        """
    self._done_adding = True