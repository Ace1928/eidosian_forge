import math
from typing import Iterator, List, Optional, Tuple
import numpy as np
from ray.data._internal.memory_tracing import trace_allocation
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef
def executed_num_blocks(self) -> int:
    """Returns the number of output blocks after execution.

        This may differ from initial_num_blocks() for LazyBlockList, which
        doesn't know how many blocks will be produced until tasks finish.
        """
    return len(self.get_blocks())