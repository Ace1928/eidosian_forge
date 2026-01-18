import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models import FactorRange, FixedTicker
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.options import AbbreviatedException, Cycle, Palette
from holoviews.element import Curve
from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_curve_padding_square(self):
    curve = Curve([1, 2, 3]).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(curve)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, -0.2)
    self.assertEqual(x_range.end, 2.2)
    self.assertEqual(y_range.start, 0.8)
    self.assertEqual(y_range.end, 3.2)