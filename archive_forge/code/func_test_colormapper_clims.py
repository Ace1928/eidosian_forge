import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_colormapper_clims(self):
    img = Image(np.array([[0, 1], [2, 3]])).opts(clims=(0, 4))
    plot = mpl_renderer.get_plot(img)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_clim(), (0, 4))