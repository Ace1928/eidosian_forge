from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_holomap(self):
    hm = HoloMap({'A': HeatMap(np.random.randint(0, 10, (100, 3))), 'B': HeatMap(np.random.randint(0, 10, (100, 3)))})
    plot = bokeh_renderer.get_plot(hm.opts(radial=True))
    self.assertIsInstance(plot, RadialHeatMapPlot)