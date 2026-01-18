import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_mpl_colormap_instance(self):
    try:
        from matplotlib import colormaps
        cmap = colormaps.get('Greys')
    except ImportError:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Greys')
    colors = process_cmap(cmap, 3, provider='matplotlib')
    self.assertEqual(colors, ['#ffffff', '#959595', '#000000'])