from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
@ngjit
def row_min_n_in_place_4d(ret, other):
    """Combine two row_min_n signed integer arrays.
    Equivalent to nanmin_n_in_place with -1 replacing NaN for missing data.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            for cat in range(ncat):
                _row_min_n_impl(ret[y, x, cat], other[y, x, cat])