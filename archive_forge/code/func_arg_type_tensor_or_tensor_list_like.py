from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
    arg = self.args_schema[arg_idx]
    is_tensor = isinstance(arg, DTensorSpec)
    if is_tensor:
        return True
    if not isinstance(arg, list):
        return False
    return all((isinstance(e, DTensorSpec) or e is None for e in arg))