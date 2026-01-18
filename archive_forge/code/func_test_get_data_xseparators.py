from itertools import product
import numpy as np
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from .test_plot import TestMPLPlot, mpl_renderer
def test_get_data_xseparators(self):
    plot = mpl_renderer.get_plot(self.element.opts(xmarks=4))
    data, style, ticks = plot.get_data(self.element, {'z': {'combined': (0, 3)}}, {})
    xseparators = data['xseparator']
    arrays = [np.array([[0.0, 0.25], [0.0, 0.5]]), np.array([[3.14159265, 0.25], [3.14159265, 0.5]])]
    self.assertEqual(xseparators, arrays)