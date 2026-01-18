import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def can_swap_in(self, seq_group: SequenceGroup) -> bool:
    blocks = self._get_physical_blocks(seq_group)
    num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
    num_free_blocks = self.gpu_allocator.get_num_free_blocks()
    num_required_blocks = len(blocks) + num_swapped_seqs
    return num_free_blocks - num_required_blocks >= self.watermark_blocks