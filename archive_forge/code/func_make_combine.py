from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned):
    arg_lk = dict(((k, v) for v, k in enumerate(bases)))
    arg_lk.update(dict(((k.reduction, v) for v, k in enumerate(bases) if isinstance(k, by))))
    base_is_where = [b.is_where() for b in bases]
    next_base_is_where = base_is_where[1:] + [False]
    calls = [(None if n else b._build_combine(d, antialias, cuda, partitioned), [arg_lk[i] for i in (b,) + t + ct]) for b, d, t, ct, n in zip(bases, dshapes, temps, combine_temps, next_base_is_where)]

    def combine(base_tuples):
        bases = tuple((np.stack(bs) for bs in zip(*base_tuples)))
        ret = []
        for is_where, (func, inds) in zip(base_is_where, calls):
            if func is None:
                continue
            call = func(*get(inds, bases))
            if is_where:
                ret.extend(call[::-1])
            else:
                ret.append(call)
        return tuple(ret)
    return combine