from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_bug_570():
    df = pd.DataFrame({'Time': [1456353642.2053893, 1456353642.2917893], 'data': [-59.4948743433377, 506.4847376716022]}, columns=['Time', 'data'])
    x_range = (1456323293.9859753, 1456374687.0009754)
    y_range = (-228.56721300380943, 460.4042291124646)
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=300, plot_width=1000)
    agg = cvs.line(df, 'Time', 'data', agg=ds.count())
    yi, xi = np.where(agg.values == 1)
    assert np.array_equal(yi, np.arange(73, 300))
    assert np.array_equal(xi, np.array([590] * len(yi)))