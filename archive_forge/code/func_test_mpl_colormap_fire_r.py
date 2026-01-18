import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_mpl_colormap_fire_r(self):
    colors = process_cmap('fire_r', 3, provider='matplotlib')
    self.assertEqual(colors, ['#ffffff', '#eb1300', '#000000'])