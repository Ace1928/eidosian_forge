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
def test_aggregate_points_categorical_zero_range(self):
    points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
    img = aggregate(points, dynamic=False, x_range=(0, 0), y_range=(0, 1), aggregator=ds.count_cat('z'), height=2)
    xs, ys = ([], [0.25, 0.75])
    params = dict(bounds=(0, 0, 0, 1), xdensity=1)
    expected = NdOverlay({'A': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params), 'B': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params), 'C': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params)}, kdims=['z'])
    self.assertEqual(img, expected)