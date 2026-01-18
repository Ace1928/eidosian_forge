from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_xmarks_data(self):
    """Test computation of xmarks data for single xmark.

        """
    self.plot.xmarks = 1
    test_mark_data = dict(ys=[(1, 1)], xs=[(0.5, 0)])
    cmp_mark_data = self.plot._get_xmarks_data(['o1'], self.seg_bins)
    for sub_list in ['xs', 'ys']:
        test_pairs = zip(test_mark_data[sub_list], cmp_mark_data[sub_list])
        for test_value, cmp_value in test_pairs:
            self.assertEqual(np.array(test_value), np.array(cmp_value))