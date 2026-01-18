from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
def get_num_blocks(self) -> int:
    return self.length // self.block_size