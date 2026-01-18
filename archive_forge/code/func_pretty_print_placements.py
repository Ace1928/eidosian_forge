from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def pretty_print_placements(self, placements):
    return ''.join([str(p) for p in placements])