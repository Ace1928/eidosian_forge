import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def make_reduction_func(name):

    @lazy_cache(name)
    def reduction_func(a, axis=None):
        a = ensure_lazy(a)
        fn = get_lib_fn(a.backend, name)
        nd = a.ndim
        if axis is None:
            return a.to(fn=fn, shape=())
        elif not hasattr(axis, '__len__'):
            axis = (axis,)
        axis = tuple((nd + i if i < 0 else i for i in axis))
        newshape = tuple((d for i, d in enumerate(shape(a)) if i not in axis))
        return a.to(fn=fn, args=(a, axis), shape=newshape)
    return reduction_func