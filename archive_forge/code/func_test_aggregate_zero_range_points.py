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
def test_aggregate_zero_range_points(self):
    p = Points([(0, 0), (1, 1)])
    agg = rasterize(p, x_range=(0, 0), y_range=(0, 1), expand=False, dynamic=False, width=2, height=2)
    img = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1), xdensity=1, vdims=[Dimension('Count', nodata=0)])
    self.assertEqual(agg, img)