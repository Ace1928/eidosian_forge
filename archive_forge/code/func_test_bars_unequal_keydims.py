import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_bars_unequal_keydims(self):
    try:
        self.assertEqual(self.bars1, self.bars3)
    except AssertionError as e:
        if not str(e) == 'Dimension names mismatched: Car occupants != Cyclists':
            raise Exception('Bars key dimension mismatch error not raised.')