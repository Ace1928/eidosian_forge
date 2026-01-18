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
def dask_cudf_DataFrame(*args, **kwargs):
    assert not kwargs.pop('geo', False)
    cdf = cudf.DataFrame.from_pandas(pd.DataFrame(*args, **kwargs), nan_as_null=False)
    return dask_cudf.from_cudf(cdf, npartitions=2)