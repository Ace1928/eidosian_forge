import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def out_wrapper(*out_names: str, exact_dtype: bool=False):
    default_out_names = ('out',)
    if len(out_names) == 0:
        out_names = default_out_names
    is_tensor = len(out_names) == 1

    def _out_wrapper(fn: Callable) -> Callable:
        """
        Adds the out parameter to a Python reference.
        """
        out_type = TensorLikeType if is_tensor else Tuple[tuple((TensorLikeType for _ in range(len(out_names))))]
        return_type = TensorLikeType if is_tensor else NamedTuple(f'return_types_{fn.__name__}', [(o, TensorLikeType) for o in out_names])
        sig = inspect.signature(fn)
        factory_kwargs = ('device', 'dtype')
        is_factory_fn = all((p in sig.parameters for p in factory_kwargs))

        @wraps(fn)
        def _fn(*args, out=None, **kwargs):
            if is_factory_fn and out is not None:
                for k in factory_kwargs:
                    out_attr = getattr(out, k)
                    if k not in kwargs:
                        kwargs[k] = out_attr
            result = fn(*args, **kwargs)
            assert isinstance(result, TensorLike) and is_tensor or (isinstance(result, Tuple) and len(result) == len(out_names))
            if out is not None:
                if is_tensor:
                    assert isinstance(out, TensorLike)
                    _maybe_resize_out(out, result.shape)
                    _safe_copy_out(copy_from=result, copy_to=out, exact_dtype=exact_dtype)
                else:
                    assert isinstance(out, Tuple)
                    torch._check_type(len(out) == len(result), lambda: f'expected tuple of {len(result)} elements but got {len(out)}')
                    for r, o in zip(result, out):
                        _maybe_resize_out(o, r.shape)
                        _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=exact_dtype)
            else:
                out = result
            return out if is_tensor else return_type(*out)
        out_param = inspect.Parameter('out', kind=inspect.Parameter.KEYWORD_ONLY, default=None, annotation=out_type)
        assert isinstance(sig.return_annotation, str) or sig.return_annotation in (sig.empty, out_type)
        params = chain(sig.parameters.values(), (out_param,))
        _fn.__signature__ = inspect.Signature(parameters=params, return_annotation=return_type)
        _fn.__annotations__ = fn.__annotations__
        _fn.__annotations__['out'] = out_type
        _fn.__annotations__['return'] = return_type
        if is_tensor and out_names != default_out_names:
            _fn.__annotations__[CustomOutParamAnnotation] = out_names[0]
        _fn._torch_decompositions_out_wrapper = f'This function is wrapped by {out_wrapper.__module__}.out_wrapper'
        return _fn
    return _out_wrapper