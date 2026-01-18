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
def nanmin_in_place(ret, other):
    """Min of 2 arrays but taking nans into account.  Could use np.nanmin but
    would need to replace zeros with nans where both arrays are nans.
    Accepts 3D (ny, nx, ncat) and 2D (ny, nx) arrays.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in nb.prange(len(ret)):
        if isnull(ret[i]):
            if not isnull(other[i]):
                ret[i] = other[i]
        elif not isnull(other[i]) and other[i] < ret[i]:
            ret[i] = other[i]