import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_empty(self):
    area = Area([])
    plot = bokeh_renderer.get_plot(area)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'], [])
    self.assertEqual(cds.data['y'], [])