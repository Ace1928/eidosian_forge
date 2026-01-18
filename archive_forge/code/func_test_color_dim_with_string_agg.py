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
def test_color_dim_with_string_agg(self, operation):
    from datashader.reductions import sum
    dmap = self.df.hvplot.scatter('x', 'y', c='number', aggregator='sum', **{operation: True})
    agg = dmap.callback.inputs[0].callback.operation.p.aggregator
    self.assertIsInstance(agg, sum)
    self.assertEqual(agg.column, 'number')