import matplotlib.pyplot as plt
import pyviz_comms as comms
from packaging.version import Version
from param import concrete_descendents
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.mpl import mpl_version
from holoviews.plotting.mpl.element import ElementPlot
from .. import option_intersections
def test_matplotlib_plot_definitions(self):
    self.assertEqual(option_intersections('matplotlib'), self.known_clashes)