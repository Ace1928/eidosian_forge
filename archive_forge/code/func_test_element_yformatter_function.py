import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_yformatter_function(self):

    def formatter(value):
        return str(value) + ' %'
    curve = Curve(range(10)).opts(yformatter=formatter)
    plot = mpl_renderer.get_plot(curve)
    yaxis = plot.handles['axis'].yaxis
    yformatter = yaxis.get_major_formatter()
    self.assertIsInstance(yformatter, FuncFormatter)