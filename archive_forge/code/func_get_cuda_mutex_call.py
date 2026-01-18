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
def get_cuda_mutex_call(lock: bool) -> str:
    func = 'cuda_mutex_lock' if lock else 'cuda_mutex_unlock'
    return f'{func}({arg_lk['_cuda_mutex']}, (y, x))'