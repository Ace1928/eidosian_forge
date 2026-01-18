import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('set.update')
def resolve_update(self, set, args, kws):
    iterable, = args
    assert not kws
    if not isinstance(iterable, types.IterableType):
        return
    dtype = iterable.iterator_type.yield_type
    unified = self.context.unify_pairs(set.dtype, dtype)
    if unified is not None:
        sig = signature(types.none, iterable)
        sig = sig.replace(recvr=set.copy(dtype=unified))
        return sig