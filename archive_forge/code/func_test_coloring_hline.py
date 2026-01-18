import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_coloring_hline(self):
    hspans = HSpans({'y0': [1, 3, 5], 'y1': [2, 4, 6]}).opts(alpha=hv.dim('y0').norm(), line_color='red', line_dash=hv.dim('y1').bin([0, 3, 6], ['dashed', 'solid']))
    plot = hv.renderer('bokeh').get_plot(hspans)
    assert plot.handles['glyph'].line_color == 'red'
    data = plot.handles['glyph_renderer'].data_source.data
    np.testing.assert_allclose(data['alpha'], [0, 0.5, 1])
    assert data['line_dash'] == ['dashed', 'solid', 'solid']