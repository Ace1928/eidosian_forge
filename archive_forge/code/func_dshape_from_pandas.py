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
def dshape_from_pandas(df):
    """Return a datashape.DataShape object given a pandas dataframe."""
    return len(df) * datashape.Record([(k, dshape_from_pandas_helper(df[k])) for k in df.columns])