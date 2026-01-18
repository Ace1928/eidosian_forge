import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('set.add')
def resolve_add(self, set, args, kws):
    item, = args
    assert not kws
    unified = self.context.unify_pairs(set.dtype, item)
    if unified is not None:
        sig = signature(types.none, unified)
        sig = sig.replace(recvr=set.copy(dtype=unified))
        return sig