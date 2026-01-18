from collections import defaultdict
from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
@cacheit
def dispatch_kinds(self, kinds, **kwargs):
    if len(kinds) == 1:
        result, = kinds
        if not isinstance(result, Kind):
            raise RuntimeError('%s is not a kind.' % result)
        return result
    for i, kind in enumerate(kinds):
        if not isinstance(kind, Kind):
            raise RuntimeError('%s is not a kind.' % kind)
        if i == 0:
            result = kind
        else:
            prev_kind = result
            t1, t2 = (type(prev_kind), type(kind))
            k1, k2 = (prev_kind, kind)
            func = self._dispatcher.dispatch(t1, t2)
            if func is None and self.commutative:
                func = self._dispatcher.dispatch(t2, t1)
                k1, k2 = (k2, k1)
            if func is None:
                result = UndefinedKind
            else:
                result = func(k1, k2)
            if not isinstance(result, Kind):
                raise RuntimeError('Dispatcher for {!r} and {!r} must return a Kind, but got {!r}'.format(prev_kind, kind, result))
    return result