import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
    """Allocate a physical slot for a new token."""
    logical_blocks = seq.logical_token_blocks
    block_table = self.block_tables[seq.seq_id]
    if len(block_table) < len(logical_blocks):
        if self.block_sliding_window and len(block_table) >= self.block_sliding_window:
            block_table.append(block_table[len(block_table) % self.block_sliding_window])
        else:
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None
    last_block = block_table[-1]
    assert last_block.device == Device.GPU
    if last_block.ref_count == 1:
        return None
    else:
        new_block = self.gpu_allocator.allocate()
        block_table[-1] = new_block
        self.gpu_allocator.free(last_block)
        return (last_block.block_number, new_block.block_number)