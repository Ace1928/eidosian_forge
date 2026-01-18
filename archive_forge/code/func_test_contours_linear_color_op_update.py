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
def test_contours_linear_color_op_update(self):
    contours = HoloMap({0: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}], vdims='color'), 1: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 5}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 2}], vdims='color')}).opts(color='color', framewise=True)
    plot = bokeh_renderer.get_plot(contours)
    cds = plot.handles['source']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    plot.update((0,))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})
    self.assertEqual(cds.data['color'], np.array([7, 3]))
    self.assertEqual(cmapper.low, 3)
    self.assertEqual(cmapper.high, 7)
    plot.update((1,))
    self.assertEqual(cds.data['color'], np.array([5, 2]))
    self.assertEqual(cmapper.low, 2)
    self.assertEqual(cmapper.high, 5)