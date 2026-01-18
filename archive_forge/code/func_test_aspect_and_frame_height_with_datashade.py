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
@parameterized.expand([('aspect',), ('data_aspect',)])
def test_aspect_and_frame_height_with_datashade(self, opt):
    plot = self.df.hvplot(x='x', y='y', frame_height=150, datashade=True, **{opt: 2})
    opts = Store.lookup_options('bokeh', plot[()], 'plot').kwargs
    self.assertEqual(opts[opt], 2)
    self.assertEqual(opts.get('frame_height'), 150)
    self.assertEqual(opts.get('height'), None)
    self.assertEqual(opts.get('frame_width'), None)