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
@spatialpandas_skip
def test_multi_poly_rasterize(self):
    poly = Polygons([{'x': [0, 1, 2, np.nan, 0, 0, 1], 'y': [0, 1, 0, np.nan, 0, 1, 1]}], datatype=['spatialpandas'])
    agg = rasterize(poly, width=4, height=4, dynamic=False)
    xs = [0.25, 0.75, 1.25, 1.75]
    ys = [0.125, 0.375, 0.625, 0.875]
    arr = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)