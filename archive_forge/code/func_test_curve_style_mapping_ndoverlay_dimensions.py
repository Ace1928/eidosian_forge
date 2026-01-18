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
def test_curve_style_mapping_ndoverlay_dimensions(self):
    ndoverlay = NdOverlay({(0, 'A'): Curve([1, 2, 0]), (0, 'B'): Curve([1, 2, 1]), (1, 'A'): Curve([1, 2, 2]), (1, 'B'): Curve([1, 2, 3])}, ['num', 'cat']).opts({'Curve': dict(color=dim('num').categorize({0: 'red', 1: 'blue'}), line_dash=dim('cat').categorize({'A': 'solid', 'B': 'dashed'}))})
    plot = bokeh_renderer.get_plot(ndoverlay)
    for (num, cat), sp in plot.subplots.items():
        glyph = sp.handles['glyph']
        color = glyph.line_color
        if num == 0:
            self.assertEqual(color, 'red')
        else:
            self.assertEqual(color, 'blue')
        linestyle = glyph.line_dash
        if cat == 'A':
            self.assertEqual(linestyle, [])
        else:
            self.assertEqual(linestyle, [6])