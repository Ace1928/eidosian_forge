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
def make_create(bases, dshapes, cuda):
    creators = [b._build_create(d) for b, d in zip(bases, dshapes)]
    if cuda:
        import cupy
        array_module = cupy
    else:
        array_module = np
    return lambda shape: tuple((c(shape, array_module) for c in creators))