import functools
import operator
from typing import cast, Dict, List, Optional, Sequence, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.distributed._tensor.tp_conv import (
from torch.distributed.device_mesh import DeviceMesh
def decompose_handler(op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> object:
    """
    Decomposes a op to core ATen op, this handler is mostly here
    for inference mode usage where the ops are not core aten ops.
    """
    r = op_call.decompose(*args, **kwargs)
    if r is not NotImplemented:
        return r
    else:
        raise RuntimeError('Decomposition failed')