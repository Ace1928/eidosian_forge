import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dmap_element_mul_element(self):
    test = self.dmap_element * self.element
    initialize_dynamic(test)
    layers = [self.dmap_element, self.element]
    self.assertEqual(split_dmap_overlay(test), layers)