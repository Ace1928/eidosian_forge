import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def free(self, seq: Sequence) -> None:
    if seq.seq_id not in self.block_tables:
        return
    block_table = self.block_tables[seq.seq_id]
    self._free_block_table(block_table)
    del self.block_tables[seq.seq_id]