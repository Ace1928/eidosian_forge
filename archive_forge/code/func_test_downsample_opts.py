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
def test_downsample_opts(self):
    plot = self.df.hvplot.line(downsample=True, width=100, height=50, x_sampling=5, xlim=(0, 5))
    assert plot.callback.operation.p.width == 100
    assert plot.callback.operation.p.height == 50
    assert plot.callback.operation.p.x_sampling == 5
    assert plot.callback.operation.p.x_range == (0, 5)