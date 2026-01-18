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
def factory_with_sizes_rule(node: fx.Node, args: Tuple[Any, ...], kwargs: Dict[str, Any], default_mesh: DeviceMesh) -> DTensor:
    flat_args = pytree.arg_tree_leaves(*args)
    assert not any((isinstance(a, DTensor) for a in flat_args)), f'Not expect DTensor argument for factory op, but got {node.target} with arguments {args}.'
    assert isinstance(args[0], list), f'Expect 2nd argument as list but got {args[1]}'
    local_sizes, placements = unpack_sizes_and_dims(args[0], default_mesh)
    node.args = (local_sizes, *args[1:])
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(local_tensor=op(*node.args, **kwargs), device_mesh=default_mesh, placements=placements, run_check=False)