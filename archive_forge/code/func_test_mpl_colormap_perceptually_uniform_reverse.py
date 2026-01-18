import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_mpl_colormap_perceptually_uniform_reverse(self):
    colors = mplcmap_to_palette('viridis_r', 4)
    self.assertEqual(colors, ['#440154', '#30678d', '#35b778', '#fde724'][::-1])