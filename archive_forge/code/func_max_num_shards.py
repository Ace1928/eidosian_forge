from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def max_num_shards(self) -> int:
    """
        Returns the max number of shards across all placement strategies
        """
    return max([strategy.output_spec.num_shards for strategy in self.strategies])