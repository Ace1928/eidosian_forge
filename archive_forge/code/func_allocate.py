import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def allocate(self, seq_group: SequenceGroup) -> None:
    seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
    num_prompt_blocks = len(seq.logical_token_blocks)
    block_table: BlockTable = []
    prefix_block_table: BlockTable = []
    num_prefix_blocks = 0
    prefix = seq_group.prefix
    if prefix is not None and prefix.allocated:
        num_prompt_blocks -= prefix.get_num_blocks()
        for block in prefix.block_table:
            block.ref_count += seq_group.num_seqs()
            block_table.append(block)
    for logical_idx in range(num_prompt_blocks):
        if self.block_sliding_window is not None and logical_idx >= self.block_sliding_window:
            block = block_table[logical_idx % self.block_sliding_window]
        else:
            block = self.gpu_allocator.allocate()
        block.ref_count = seq_group.num_seqs()
        block_table.append(block)
    if prefix is not None and (not prefix.allocated):
        num_prefix_blocks = prefix.get_num_blocks()
        prefix_block_table = block_table[:num_prefix_blocks]
        for block in prefix_block_table:
            block.ref_count += 1
        prefix.set_block_table(prefix_block_table)
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        self.block_tables[seq.seq_id] = block_table.copy()