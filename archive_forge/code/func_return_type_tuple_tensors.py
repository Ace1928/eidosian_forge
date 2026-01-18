from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def return_type_tuple_tensors(self) -> bool:
    return_types = self.op._schema.returns
    return len(return_types) > 1 and isinstance(return_types[0].type, torch.TensorType)