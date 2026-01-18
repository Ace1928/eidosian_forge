import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
def dispatch_functionalize(func):

    def to_fun(t):
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        return t

    def from_fun(t):
        if not isinstance(t, FunctionalTensor):
            if isinstance(t, torch.Tensor):
                assert not torch._is_functional_tensor(t)
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t.elem)

    def inner(*args, **kwargs):
        func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
        func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)
        flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
        flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            func_outputs = func(*func_args, **func_kwargs)
            outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)
            return outputs
    return inner