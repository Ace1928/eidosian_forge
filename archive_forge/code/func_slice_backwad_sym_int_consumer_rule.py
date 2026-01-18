import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.op_schema import OpSchema
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def slice_backwad_sym_int_consumer_rule(node: fx.Node, args: Tuple[Any, ...]) -> DTensor:
    grad_output, input_sizes, dim, start, end, step = args
    local_sizes: List[int] = [s.local_value if isinstance(s, DSymInt) else s for s in input_sizes]
    input_tensor = torch.zeros(local_sizes, device=grad_output.device, dtype=grad_output.dtype)
    return DTensor.from_local(local_tensor=torch.slice_scatter(input_tensor, grad_output.to_local(), dim, start, end, step), device_mesh=grad_output.device_mesh, placements=grad_output.placements, run_check=False)