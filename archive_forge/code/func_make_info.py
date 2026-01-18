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
def make_info(cols, cuda, uses_cuda_mutex: bool):

    def info(df, canvas_shape):
        ret = tuple((c.apply(df, cuda) for c in cols))
        if uses_cuda_mutex:
            import cupy
            import numba
            from packaging.version import Version
            if Version(numba.__version__) >= Version('0.57'):
                mutex_array = cupy.zeros(canvas_shape, dtype=np.uint32)
            else:
                mutex_array = cupy.zeros((1,), dtype=np.uint32)
            ret += (mutex_array,)
        return ret
    return info