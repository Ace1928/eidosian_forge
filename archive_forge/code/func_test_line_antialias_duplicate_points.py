from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('self_intersect', [False, True])
def test_line_antialias_duplicate_points(self_intersect):
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(-0.1, 1.1), y_range=(0.9, 2.1))
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2]))
    agg_no_duplicate = cvs.line(source=df, x='x', y='y', line_width=1, agg=ds.count(self_intersect=self_intersect))
    df = pd.DataFrame(dict(x=[0, 0, 1], y=[1, 1, 2]))
    agg_duplicate = cvs.line(source=df, x='x', y='y', line_width=1, agg=ds.count(self_intersect=self_intersect))
    assert_eq_xr(agg_no_duplicate, agg_duplicate)