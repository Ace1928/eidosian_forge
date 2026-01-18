import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_labels_plot_rotation_style(self):
    text = Labels([(0, 0, 'Test')]).opts(angle=90)
    plot = bokeh_renderer.get_plot(text)
    glyph = plot.handles['glyph']
    self.assertEqual(glyph.angle, np.pi / 2.0)