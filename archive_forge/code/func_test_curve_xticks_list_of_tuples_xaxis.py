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
def test_curve_xticks_list_of_tuples_xaxis(self):
    ticks = [(0, 'zero'), (5, 'five'), (10, 'ten')]
    curve = Curve(range(10)).opts(xticks=ticks)
    plot = bokeh_renderer.get_plot(curve).state
    self.assertIsInstance(plot.xaxis[0].ticker, FixedTicker)
    self.assertEqual(plot.xaxis[0].major_label_overrides, dict(ticks))