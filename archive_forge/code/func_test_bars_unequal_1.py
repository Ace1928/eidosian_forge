import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_bars_unequal_1(self):
    try:
        self.assertEqual(self.bars1, self.bars2)
    except AssertionError as e:
        if 'not almost equal' not in str(e):
            raise Exception(f'Bars mismatched data error not raised. {e}')