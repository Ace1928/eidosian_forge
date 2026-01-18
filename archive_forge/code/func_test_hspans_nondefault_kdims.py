import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hspans_nondefault_kdims(self):
    hspans = HSpans({'other0': [0, 3, 5.5], 'other1': [1, 4, 6.5]}, kdims=['other0', 'other1'])
    plot = bokeh_renderer.get_plot(hspans)
    assert isinstance(plot.handles['glyph'], BkHStrip)
    assert plot.handles['xaxis'].axis_label == 'x'
    assert plot.handles['yaxis'].axis_label == 'y'
    assert plot.handles['x_range'].start == 0
    assert plot.handles['x_range'].end == 1
    assert plot.handles['y_range'].start == 0
    assert plot.handles['y_range'].end == 6.5
    source = plot.handles['source']
    assert list(source.data) == ['other0', 'other1']
    assert (source.data['other0'] == [0, 3, 5.5]).all()
    assert (source.data['other1'] == [1, 4, 6.5]).all()