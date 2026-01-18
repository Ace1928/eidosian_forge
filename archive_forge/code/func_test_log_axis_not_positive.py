from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
@pytest.mark.parametrize('canvas', [ds.Canvas(x_axis_type='log'), ds.Canvas(x_axis_type='log', x_range=(0, 1)), ds.Canvas(y_axis_type='log'), ds.Canvas(y_axis_type='log', y_range=(0, 1))])
def test_log_axis_not_positive(df, canvas):
    with pytest.raises(ValueError, match='Range values must be >0 for logarithmic axes'):
        canvas.line(df, 'x', 'y')