import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_hover_ensure_vdims_sanitized(self):
    hm = HeatMap([(1, 1, 1), (2, 2, 0)], vdims=['z with $pace']).opts(tools=['hover'])
    self._test_hover_info(hm, [('x', '@{x}'), ('y', '@{y}'), ('z with $pace', '@{z_with_pace}')])