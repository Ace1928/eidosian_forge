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
def test_curve_categorical_xaxis(self):
    curve = Curve((['A', 'B', 'C'], [1, 2, 3]))
    plot = bokeh_renderer.get_plot(curve)
    x_range = plot.handles['x_range']
    self.assertIsInstance(x_range, FactorRange)
    self.assertEqual(x_range.factors, ['A', 'B', 'C'])