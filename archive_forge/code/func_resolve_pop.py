import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('list.pop')
def resolve_pop(self, list, args, kws):
    assert not kws
    if not args:
        return signature(list.dtype)
    else:
        idx, = args
        if isinstance(idx, types.Integer):
            return signature(list.dtype, types.intp)