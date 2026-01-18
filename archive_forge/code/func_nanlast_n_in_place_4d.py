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
@ngjit_parallel
def nanlast_n_in_place_4d(ret, other):
    """3d version of nanfirst_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanlast_n_impl(ret[y, x, cat], other[y, x, cat])