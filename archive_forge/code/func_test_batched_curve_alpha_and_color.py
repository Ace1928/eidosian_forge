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
def test_batched_curve_alpha_and_color(self):
    opts = {'NdOverlay': dict(legend_limit=0), 'Curve': dict(alpha=Cycle(values=[0.5, 1]))}
    overlay = NdOverlay({i: Curve([(i, j) for j in range(2)]) for i in range(2)}).opts(opts)
    plot = bokeh_renderer.get_plot(overlay).subplots[()]
    alpha = [0.5, 1.0]
    color = ['#30a2da', '#fc4f30']
    self.assertEqual(plot.handles['source'].data['alpha'], alpha)
    self.assertEqual(plot.handles['source'].data['color'], color)