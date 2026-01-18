import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_mpl_colormap_sequential_reverse(self):
    colors = mplcmap_to_palette('YlGn_r', 3)
    self.assertEqual(colors, ['#ffffe5', '#78c679', '#004529'][::-1])