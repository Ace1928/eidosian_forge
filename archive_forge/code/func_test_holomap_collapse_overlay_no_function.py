import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_holomap_collapse_overlay_no_function(self):
    hmap = HoloMap({(1, 0): Curve(np.arange(8)) * Curve(-np.arange(8)), (2, 0): Curve(np.arange(8) ** 2) * Curve(-np.arange(8) ** 3)}, kdims=['A', 'B'])
    self.assertEqual(hmap.collapse(), Overlay([(('Curve', 'I'), Dataset({'A': np.concatenate([np.ones(8), np.ones(8) * 2]), 'B': np.zeros(16), 'x': np.tile(np.arange(8), 2), 'y': np.concatenate([np.arange(8), np.arange(8) ** 2])}, kdims=['A', 'B', 'x'], vdims=['y'])), (('Curve', 'II'), Dataset({'A': np.concatenate([np.ones(8), np.ones(8) * 2]), 'B': np.zeros(16), 'x': np.tile(np.arange(8), 2), 'y': np.concatenate([-np.arange(8), -np.arange(8) ** 3])}, kdims=['A', 'B', 'x'], vdims=['y']))]))