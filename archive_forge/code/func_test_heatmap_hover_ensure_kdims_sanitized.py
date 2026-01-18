import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_hover_ensure_kdims_sanitized(self):
    hm = HeatMap([(1, 1, 1), (2, 2, 0)], kdims=['x with space', 'y with $pecial symbol'])
    hm = hm.opts(tools=['hover'])
    self._test_hover_info(hm, [('x with space', '@{x_with_space}'), ('y with $pecial symbol', '@{y_with_pecial_symbol}'), ('z', '@{z}')])