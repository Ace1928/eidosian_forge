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
def isminus1(val):
    """
    Check for -1 which is equivalent to NaN for some integer aggregations
    """
    return val == -1