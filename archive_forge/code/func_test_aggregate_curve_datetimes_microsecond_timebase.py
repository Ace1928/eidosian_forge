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
def test_aggregate_curve_datetimes_microsecond_timebase(self):
    dates = pd.date_range(start='2016-01-01', end='2016-01-03', freq='1D')
    xstart = np.datetime64('2015-12-31T23:59:59.723518000', 'us')
    xend = np.datetime64('2016-01-03T00:00:00.276482000', 'us')
    curve = Curve((dates, [1, 2, 3]))
    img = aggregate(curve, width=2, height=2, x_range=(xstart, xend), dynamic=False)
    bounds = (np.datetime64('2015-12-31T23:59:59.723518'), 1.0, np.datetime64('2016-01-03T00:00:00.276482'), 3.0)
    dates = [np.datetime64('2016-01-01T11:59:59.861759000'), np.datetime64('2016-01-02T12:00:00.138241000')]
    expected = Image((dates, [1.5, 2.5], [[1, 0], [0, 2]]), datatype=['xarray'], bounds=bounds, vdims=Dimension('Count', nodata=0))
    self.assertEqual(img, expected)