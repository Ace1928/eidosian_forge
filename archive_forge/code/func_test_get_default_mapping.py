from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_default_mapping(self):
    glyphs = self.plot._style_groups.keys()
    glyphs_mapped = self.plot.get_default_mapping(None, None).keys()
    glyphs_plain = {x[:-2] for x in glyphs_mapped}
    self.assertTrue(all([x in glyphs_plain for x in glyphs]))