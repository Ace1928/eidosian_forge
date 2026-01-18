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
def test_curve_overlay_datetime_hover(self):
    obj = NdOverlay({i: Curve([(dt.datetime(2016, 1, j + 1), j) for j in range(31)]) for i in range(5)}, kdims=['Test'])
    opts = {'Curve': {'tools': ['hover']}}
    obj = obj.opts(opts)
    self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}{%F %T}'), ('y', '@{y}')], formatters={'@{x}': 'datetime'})