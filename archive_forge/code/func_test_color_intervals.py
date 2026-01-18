import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_color_intervals(self):
    levels = [0, 38, 73, 95, 110, 130, 156]
    colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20']
    cmap, lims = color_intervals(colors, levels, N=10)
    self.assertEqual(cmap, ['#5ebaff', '#5ebaff', '#00faf4', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff8f20'])