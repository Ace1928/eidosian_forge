import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
def once_differentiable(fn):

    @functools.wraps(fn)
    def wrapper(ctx, *args):
        with torch.no_grad():
            outputs = fn(ctx, *args)
        if not torch.is_grad_enabled():
            return outputs
        requires_grad = any((isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args))
        if not requires_grad:
            return outputs
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        err_fn = _functions.DelayedError(b'trying to differentiate twice a function that was marked with @once_differentiable', len(outputs))

        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var
        return err_fn(*[fake_requires_grad(v) for v in outputs])
    return wrapper