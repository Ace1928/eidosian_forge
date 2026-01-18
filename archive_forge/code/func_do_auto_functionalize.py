from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
def do_auto_functionalize(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, mutated_args_names, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI
    ctx = PythonFunctionalizeAPI()
    mutable_args_names = []
    normalized_kwargs = {}
    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        if arg.alias_info is not None and arg.alias_info.is_write:
            mutable_args_names.append(arg.name)
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        else:
            normalized_kwargs[arg.name] = args[idx]
    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)
    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized(op, mutable_args_names, unwrapped_kwargs)
    assert len(unwrapped_outs) == len(mutable_args_names)
    for name, unwrapped_out in zip(mutable_args_names, unwrapped_outs):
        if unwrapped_out is None:
            continue
        assert isinstance(unwrapped_out, torch.Tensor)
        orig_arg = normalized_kwargs[name]
        ctx.replace(orig_arg, unwrapped_out)
        ctx.commit_update(orig_arg)
        ctx.sync(orig_arg)
    return None