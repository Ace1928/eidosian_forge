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
def binop_sym_int_consumer_rule(node: fx.Node, args: Tuple[Any, ...]) -> DTensor:
    assert len(args) == 2, f'Expect two args but got op {node.target} with args {args}'
    assert isinstance(args[0], DTensor), f'Expect 1st argument to be DTensor but got {args[0]}'
    assert isinstance(args[1], list), f'Expect 2nd argument as list but got {args[1]}'
    local_sizes, placements = unpack_sizes_and_dims(args[1], args[0].device_mesh)
    node.args = (node.args[0], local_sizes)
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(local_tensor=op(args[0]._local_tensor, local_sizes), device_mesh=args[0].device_mesh, placements=placements, run_check=False)