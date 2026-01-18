from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalizer_inner(func):

    @functools.wraps(func)
    def wrapped(*args, **kwds):
        sig = inspect.signature(func)
        params = sig.parameters
        first_param = next(iter(params.values()))
        if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
            args = [maybe_normalize(arg, first_param) for arg in args]
        else:
            args = tuple((maybe_normalize(arg, parm) for arg, parm in zip(args, params.values()))) + args[len(params.values()):]
        kwds = {name: maybe_normalize(arg, params[name]) if name in params else arg for name, arg in kwds.items()}
        result = func(*args, **kwds)
        bound_args = None
        if 'keepdims' in params and params['keepdims'].annotation == 'KeepDims':
            bound_args = sig.bind(*args, **kwds).arguments
            if bound_args.get('keepdims', False):
                tensor = args[0]
                axis = bound_args.get('axis')
                result = _util.apply_keepdims(result, axis, tensor.ndim)
        if 'out' in params:
            if bound_args is None:
                bound_args = sig.bind(*args, **kwds).arguments
            out = bound_args.get('out')
            result = maybe_copy_to(out, result, promote_scalar_result)
        result = wrap_tensors(result)
        return result
    return wrapped