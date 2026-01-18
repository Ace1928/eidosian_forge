import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_colorbar_empty_clabel(self):
    img = Image(np.array([[1, 1, 1, 2], [2, 2, 3, 4]])).opts(clabel='', colorbar=True)
    plot = mpl_renderer.get_plot(img)
    colorbar = plot.handles['cax']
    self.assertEqual(colorbar.get_label(), '')