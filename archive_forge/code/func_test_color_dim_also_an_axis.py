import sys
from unittest import SkipTest
from parameterized import parameterized
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
from holoviews import Store, render
from holoviews.element import Image, QuadMesh, Points
from holoviews.core.spaces import DynamicMap
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Scatter
from holoviews.element.comparison import ComparisonTestCase
from hvplot.converter import HoloViewsConverter
from hvplot.tests.util import makeTimeDataFrame
from packaging.version import Version
@parameterized.expand([('rasterize',), ('datashade',)])
def test_color_dim_also_an_axis(self, operation):
    from datashader.reductions import mean
    original_data = self.df.copy(deep=True)
    dmap = self.df.hvplot.scatter('x', 'y', c='y', **{operation: True})
    agg = dmap.callback.inputs[0].callback.operation.p.aggregator
    self.assertIsInstance(agg, mean)
    self.assertEqual(agg.column, '_color')
    assert original_data.equals(self.df)