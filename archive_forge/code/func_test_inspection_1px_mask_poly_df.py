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
def test_inspection_1px_mask_poly_df(self):
    if spatialpandas is None:
        raise SkipTest('Polygon inspect tests require spatialpandas')
    inspector = inspect.instance(max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
    inspector(self.polysrgb)
    self.assertEqual(len(inspector.hits), 1)
    data = [[6.0, 7.0, 3.0, 2.0, 7.0, 5.0, 6.0, 7.0]]
    self.assertEqual(inspector.hits.iloc[0].geometry, spatialpandas.geometry.polygon.Polygon(data))