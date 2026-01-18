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
@pytest.mark.parametrize('ddf', ddfs)
@pytest.mark.parametrize('reduction,dtype,aa_dtype', [(ds.any(), bool, np.float32), (ds.count(), np.uint32, np.float32), (ds.max('f64'), np.float64, np.float64), (ds.min('f64'), np.float64, np.float64), (ds.sum('f64'), np.float64, np.float64)])
def test_combine_dtype(ddf, reduction, dtype, aa_dtype):
    if dask_cudf and isinstance(ddf, dask_cudf.DataFrame):
        pytest.skip('antialiased lines not supported with cudf')
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    agg = cvs.line(ddf, 'x', 'y', line_width=0, agg=reduction)
    assert agg.dtype == dtype
    agg = cvs.line(ddf, 'x', 'y', line_width=1, agg=reduction)
    assert agg.dtype == aa_dtype