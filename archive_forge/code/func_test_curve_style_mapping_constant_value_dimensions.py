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
def test_curve_style_mapping_constant_value_dimensions(self):
    vdims = ['y', 'num', 'cat']
    ndoverlay = NdOverlay({0: Curve([(0, 1, 0, 'A'), (1, 0, 0, 'A')], vdims=vdims), 1: Curve([(0, 1, 0, 'B'), (1, 1, 0, 'B')], vdims=vdims), 2: Curve([(0, 1, 1, 'A'), (1, 2, 1, 'A')], vdims=vdims), 3: Curve([(0, 1, 1, 'B'), (1, 3, 1, 'B')], vdims=vdims)}).opts({'Curve': dict(color=dim('num').categorize({0: 'red', 1: 'blue'}), line_dash=dim('cat').categorize({'A': 'solid', 'B': 'dashed'}))})
    plot = bokeh_renderer.get_plot(ndoverlay)
    for k, sp in plot.subplots.items():
        glyph = sp.handles['glyph']
        color = glyph.line_color
        if ndoverlay[k].iloc[0, 2] == 0:
            self.assertEqual(color, 'red')
        else:
            self.assertEqual(color, 'blue')
        linestyle = glyph.line_dash
        if ndoverlay[k].iloc[0, 3] == 'A':
            self.assertEqual(linestyle, [])
        else:
            self.assertEqual(linestyle, [6])