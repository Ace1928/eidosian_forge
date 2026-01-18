from typing import Generic
from ray.data.block import Block, BlockAccessor, T
@staticmethod
def for_block(block: Block) -> 'BlockBuilder':
    return BlockAccessor.for_block(block).builder()