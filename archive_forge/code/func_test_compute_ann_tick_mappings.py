from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_compute_ann_tick_mappings(self):
    """Test computation of annular tick mappings. Check integers, list and
        function types.

        """
    order = sorted(self.ann_bins.keys())
    self.plot.yticks = 1
    ticks = self.plot._compute_tick_mapping('radius', order, self.ann_bins)
    self.assertEqual(ticks, {'o1': self.ann_bins['o1']})
    self.plot.yticks = 2
    ticks = self.plot._compute_tick_mapping('radius', order, self.ann_bins)
    self.assertEqual(ticks, self.ann_bins)
    self.plot.yticks = ['New Tick1', 'New Tick2']
    ticks = self.plot._compute_tick_mapping('radius', order, self.ann_bins)
    bins = self.plot._get_bins('radius', self.plot.yticks)
    ticks_cmp = {x: bins[x] for x in self.plot.yticks}
    self.assertEqual(ticks, ticks_cmp)
    self.plot.yticks = lambda x: x == 'o1'
    ticks = self.plot._compute_tick_mapping('radius', order, self.ann_bins)
    self.assertEqual(ticks, {'o1': self.ann_bins['o1']})