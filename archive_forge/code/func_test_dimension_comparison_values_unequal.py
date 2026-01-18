from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_values_unequal(self):
    try:
        self.assertEqual(self.dimension4, self.dimension8)
    except AssertionError as e:
        self.assertEqual(str(e), "Dimension parameter 'values' mismatched: [] != ['a', 'b']")