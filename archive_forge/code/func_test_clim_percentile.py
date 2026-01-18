import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_clim_percentile(self):
    arr = np.random.rand(10, 10)
    arr[0, 0] = -100
    arr[-1, -1] = 100
    im = Image(arr).opts(clim_percentile=True)
    plot = bokeh_renderer.get_plot(im)
    low, high = plot.ranges['Image',]['z']['robust']
    assert low > 0
    assert high < 1