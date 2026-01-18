from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_angle_bin_computation(self):
    """Test computation of bins for radiants/segments.

        """
    order = sorted(self.seg_bins.keys())
    values = self.plot._get_bins('angle', order)
    self.assertEqual(values.keys(), self.seg_bins.keys())
    self.assertEqual(values['o1'], self.seg_bins['o1'])
    self.assertEqual(values['o2'], self.seg_bins['o2'])
    values = self.plot._get_bins('angle', order, True)
    self.assertEqual(values.keys(), self.seg_bins.keys())
    self.assertEqual(values['o1'], self.seg_bins['o2'])
    self.assertEqual(values['o2'], self.seg_bins['o1'])