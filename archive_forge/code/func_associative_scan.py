from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def associative_scan(input, axis, combine_fn, _builder=None, _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along the provided :code:`axis` and update the carry

    :param input: the input tensor, or tuple of tensors
    :param axis: the dimension along which the reduction should be done
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)

    """
    if isinstance(input, tensor):
        return associative_scan((input,), axis, combine_fn, _builder=_builder, _generator=_generator)[0]

    def make_combine_region(scan_op):
        in_scalar_tys = [t.type.scalar for t in input]
        prototype = function_type(in_scalar_tys, in_scalar_tys * 2)
        region = scan_op.get_region(0)
        with _insertion_guard(_builder):
            param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
            block = _builder.create_block_with_parent(region, param_types)
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(prototype.param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_scan_ret(*handles)
    axis = _constexpr_to_value(axis)
    return semantic.associative_scan(input, axis, make_combine_region, _builder)