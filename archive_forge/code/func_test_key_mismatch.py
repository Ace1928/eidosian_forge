import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
def test_key_mismatch(self):
    try:
        self.assertEqual(self.map1_1D, self.map2_1D)
        raise AssertionError('Mismatch in map keys not raised.')
    except AssertionError as e:
        self.assertEqual(str(e), 'HoloMaps have different sets of keys. In first, not second [0]. In second, not first: [2].')