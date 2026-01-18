import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_contours_empty_path(self):
    contours = Contours([pd.DataFrame([], columns=['x', 'y', 'color', 'line_width']), pd.DataFrame({'x': np.random.rand(10), 'y': np.random.rand(10), 'color': ['red'] * 10, 'line_width': [3] * 10}, columns=['x', 'y', 'color', 'line_width'])], vdims=['color', 'line_width']).opts(color='color', line_width='line_width')
    plot = bokeh_renderer.get_plot(contours)
    glyph = plot.handles['glyph']
    self.assertEqual(glyph.line_color, 'red')
    self.assertEqual(glyph.line_width, 3)