import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hlines_xlabel_ylabel(self):
    hlines = HLines({'y': [0, 1, 2, 5.5], 'extra': [-1, -2, -3, -44]}, vdims=['extra']).opts(xlabel='xlabel', ylabel='xlabel')
    plot = bokeh_renderer.get_plot(hlines)
    assert isinstance(plot.handles['glyph'], BkHSpan)
    assert plot.handles['xaxis'].axis_label == 'xlabel'
    assert plot.handles['yaxis'].axis_label == 'xlabel'