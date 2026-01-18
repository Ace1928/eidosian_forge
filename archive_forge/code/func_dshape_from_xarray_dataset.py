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
def dshape_from_xarray_dataset(xr_ds):
    """Return a datashape.DataShape object given a xarray Dataset."""
    return datashape.var * datashape.Record([(k, dshape_from_pandas_helper(xr_ds[k])) for k in list(xr_ds.data_vars) + list(xr_ds.coords)])