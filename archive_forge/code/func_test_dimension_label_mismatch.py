import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_label_mismatch(self):
    try:
        self.assertEqual(self.map1_1D, self.map6_1D)
        raise AssertionError('Mismatch in dimension labels not raised.')
    except AssertionError as e:
        self.assertEqual(str(e), 'Dimension names mismatched: int != int_v2')