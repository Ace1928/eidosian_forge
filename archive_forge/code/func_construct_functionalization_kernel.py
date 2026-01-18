import weakref
import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
from .autograd import autograd_not_implemented
def construct_functionalization_kernel(mutable_op, functional_op):

    def kernel(*args):
        if pytree.tree_all_only(torch.Tensor, lambda x: not torch._is_functional_tensor(x), args):
            with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
                return mutable_op(*args)
        if not pytree.tree_all_only(torch.Tensor, torch._is_functional_tensor, args):
            raise RuntimeError('{mutable_op}: expected all args to be FunctionalTensorWrapper')
        unwrapped_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch._is_functional_tensor(arg):
                torch._sync(arg)
                unwrapped = torch._from_functional_tensor(arg)
                unwrapped_args.append(unwrapped)
            else:
                unwrapped_args.append(arg)
        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
            output = functional_op(*unwrapped_args)
        num_actual_output = len(mutable_op._schema.returns)
        actual_output = pytree.tree_map(torch._to_functional_tensor, output[:num_actual_output])
        new_values_to_propagate = output[num_actual_output:]
        inputs_to_replace = [arg for is_write, arg in zip(mutable_args(mutable_op), args) if is_write]
        assert len(new_values_to_propagate) == len(inputs_to_replace)
        for new_value, arg in zip(new_values_to_propagate, inputs_to_replace):
            if arg is None and new_value is None or (arg is not None and new_value is not None):
                continue
            torch._C._propagate_xla_data(arg, new_value)
            torch._C._replace_(arg, new_value)
            torch._C._commit_update(arg)
            torch._sync(arg)
        if len(actual_output) == 1:
            return actual_output[0]
        elif len(actual_output) == 0:
            return None
        return actual_output
    return kernel