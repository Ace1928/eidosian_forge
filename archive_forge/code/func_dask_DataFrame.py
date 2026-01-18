from __future__ import annotations
import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.context import config
from numpy import nan
import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du
import pytest
from datashader.tests.test_pandas import (
def dask_DataFrame(*args, **kwargs):
    if kwargs.pop('geo', False):
        df = sp.GeoDataFrame(*args, **kwargs)
    else:
        df = pd.DataFrame(*args, **kwargs)
    return dd.from_pandas(df, npartitions=2)