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
def test_area_aggregate_crossover_count(self):
    area = Area([-1, 2, 3])
    agg = rasterize(area, width=4, height=4, y_range=(-3, 3), dynamic=False)
    xs = [0.25, 0.75, 1.25, 1.75]
    ys = [-2.25, -0.75, 0.75, 2.25]
    arr = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)