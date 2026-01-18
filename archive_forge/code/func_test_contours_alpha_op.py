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
def test_contours_alpha_op(self):
    contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}], vdims='alpha').opts(alpha='alpha')
    plot = bokeh_renderer.get_plot(contours)
    cds = plot.handles['source']
    glyph = plot.handles['glyph']
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})
    self.assertEqual(cds.data['alpha'], np.array([0.7, 0.3]))