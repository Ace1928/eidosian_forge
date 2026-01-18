import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_zformatter_instance(self):
    formatter = PercentFormatter()
    curve = Scatter3D([]).opts(zformatter=formatter)
    plot = mpl_renderer.get_plot(curve)
    zaxis = plot.handles['axis'].zaxis
    zformatter = zaxis.get_major_formatter()
    self.assertIs(zformatter, formatter)