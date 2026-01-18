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
def row_min_in_place(ret, other):
    """Minimum of 2 arrays of row indexes.
    Row indexes are integers from 0 upwards, missing data is -1.
    Return the first array.
    """
    ret = ret.ravel()
    other = other.ravel()
    for i in range(len(ret)):
        if other[i] > -1 and (ret[i] == -1 or other[i] < ret[i]):
            ret[i] = other[i]