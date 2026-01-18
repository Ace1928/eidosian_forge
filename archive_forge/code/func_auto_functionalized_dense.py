from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(op: torch._ops.OpOverload, mutated_args_names: List[str], kwargs: Dict[str, Any]) -> Tuple[Tensor, ...]:
    new_kwargs = dict(**kwargs)
    result = []
    for name in mutated_args_names:
        new_kwargs[name] = clone_preserve_strides(kwargs[name]) if kwargs[name] is not None else None
        result.append(new_kwargs[name])
    out = op(**new_kwargs)
    assert out is None
    return tuple(result)