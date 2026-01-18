import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_bokeh_palette_diverging(self):
    colors = bokeh_palette_to_palette('RdBu', 3)
    self.assertEqual(colors, ['#67001f', '#f7f7f7', '#053061'])