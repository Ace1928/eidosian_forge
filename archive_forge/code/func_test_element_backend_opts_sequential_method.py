import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_backend_opts_sequential_method(self):
    a = Curve([1, 2, 3], label='a')
    b = Curve([1, 4, 9], label='b')
    curve = (a * b).opts(show_legend=True, backend_opts={'legend.get_title().set_fontsize': 188})
    plot = mpl_renderer.get_plot(curve)
    legend = plot.handles['legend']
    self.assertEqual(legend.get_title().get_fontsize(), 188)