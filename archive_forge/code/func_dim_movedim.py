from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
def dim_movedim(ndim: int, input: Union[int, Sequence[int]], destination: Union[int, Sequence[int]]) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)
    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), 'Found repeated input dims'
    assert len(set(destination)) == len(destination), 'Found repeated output dims'
    assert max(input) < ndim
    assert max(destination) < ndim
    dest = [-1] * ndim
    for i, d in zip(input, destination):
        dest[d] = i
    unused_inputs_iter = iter((i for i in range(ndim) if i not in input_set))
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)
    return tuple((InputDim(i) for i in dest))