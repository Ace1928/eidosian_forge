from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_ymarks_data(self):
    """Test computation of ymarks data for single ymark.

        """
    self.plot.ymarks = 1
    test_mark_data = dict(radius=self.ann_bins['o1'][1])
    cmp_mark_data = self.plot._get_ymarks_data(['o1'], self.ann_bins)
    self.assertEqual(test_mark_data, cmp_mark_data)