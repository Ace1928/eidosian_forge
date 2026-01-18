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
def test_batched_curve_line_color_and_color(self):
    opts = {'NdOverlay': dict(legend_limit=0), 'Curve': dict(line_color=Cycle(values=['red', 'blue']))}
    overlay = NdOverlay({i: Curve([(i, j) for j in range(2)]) for i in range(2)}).opts(opts)
    plot = bokeh_renderer.get_plot(overlay).subplots[()]
    line_color = ['red', 'blue']
    self.assertEqual(plot.handles['source'].data['line_color'], line_color)