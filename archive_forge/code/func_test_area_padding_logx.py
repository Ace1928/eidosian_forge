import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_padding_logx(self):
    area = Area([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
    plot = bokeh_renderer.get_plot(area)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.8959584598407623)
    self.assertEqual(x_range.end, 3.348369522101713)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)