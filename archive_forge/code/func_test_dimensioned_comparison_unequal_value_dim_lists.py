from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimensioned_comparison_unequal_value_dim_lists(self):
    try:
        self.assertEqual(self.dimensioned1, self.dimensioned4)
    except AssertionError as e:
        self.assertEqual(str(e), 'Value dimension list mismatched')