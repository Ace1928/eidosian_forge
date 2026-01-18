from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('reduction', [ds.std('value'), ds.var('value')])
def test_line_antialias_reduction_not_implemented(reduction):
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2]))
    with pytest.raises(NotImplementedError):
        cvs.line(df, 'x', 'y', line_width=1, agg=reduction)