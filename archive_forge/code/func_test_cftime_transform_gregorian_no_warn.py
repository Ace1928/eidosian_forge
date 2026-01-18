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
def test_cftime_transform_gregorian_no_warn(self):
    try:
        import cftime
    except ImportError:
        raise SkipTest('Test requires cftime library')
    gregorian_dates = [cftime.DatetimeGregorian(2000, 2, 28), cftime.DatetimeGregorian(2000, 3, 1), cftime.DatetimeGregorian(2000, 3, 2)]
    curve = Curve((gregorian_dates, [1, 2, 3]))
    plot = bokeh_renderer.get_plot(curve)
    xs = plot.handles['cds'].data['x']
    self.assertEqual(xs.astype('int64'), np.array([951696000000, 951868800000, 951955200000]))