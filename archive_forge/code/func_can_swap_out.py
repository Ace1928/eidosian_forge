import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def can_swap_out(self, seq_group: SequenceGroup) -> bool:
    blocks = self._get_physical_blocks(seq_group)
    return len(blocks) <= self.cpu_allocator.get_num_free_blocks()