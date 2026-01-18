from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_ann_labels_data(self):
    """Test correct computation of a single annular label data point.

        """
    test_ann_data = dict(x=np.array(1), y=np.array(self.ann_bins['o1'][0] + 1), text=np.array('o1'), angle=[0])
    cmp_ann_data = self.plot._get_ann_labels_data(['o1'], self.ann_bins)
    self.assertEqual(test_ann_data, cmp_ann_data)