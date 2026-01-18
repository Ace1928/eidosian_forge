import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_xformatter_function(self):

    def formatter(value):
        return str(value) + ' %'
    curve = Curve(range(10)).opts(xformatter=formatter)
    plot = mpl_renderer.get_plot(curve)
    xaxis = plot.handles['axis'].xaxis
    xformatter = xaxis.get_major_formatter()
    self.assertIsInstance(xformatter, FuncFormatter)