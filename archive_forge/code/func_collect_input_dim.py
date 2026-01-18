from typing import Callable, Dict, List, Set
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
def collect_input_dim(cmd: DimSpec, input_dims: Set[int]):
    if isinstance(cmd, InputDim):
        input_dims.add(cmd.input_dim)
    for inp in cmd.inputs():
        collect_input_dim(inp, input_dims)