import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_padding_nonsquare(self):
    area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(area)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.9)
    self.assertEqual(x_range.end, 3.1)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)