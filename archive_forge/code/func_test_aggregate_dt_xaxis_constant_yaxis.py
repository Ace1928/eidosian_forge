import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
def test_aggregate_dt_xaxis_constant_yaxis(self):
    df = pd.DataFrame({'y': np.ones(100)}, index=pd.date_range('1980-01-01', periods=100, freq='1min'))
    img = rasterize(Curve(df), dynamic=False, width=3)
    xs = np.array(['1980-01-01T00:16:30.000000', '1980-01-01T00:49:30.000000', '1980-01-01T01:22:30.000000'], dtype='datetime64[us]')
    ys = np.array([])
    bounds = (np.datetime64('1980-01-01T00:00:00.000000'), 1.0, np.datetime64('1980-01-01T01:39:00.000000'), 1.0)
    expected = Image((xs, ys, np.empty((0, 3))), ['index', 'y'], vdims=Dimension('Count', nodata=0), xdensity=1, ydensity=1, bounds=bounds)
    self.assertEqual(img, expected)