from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimensioned_comparison_unequal_value_dims(self):
    try:
        self.assertEqual(self.dimensioned1, self.dimensioned2)
    except AssertionError as e:
        self.assertEqual(str(e), 'Dimension names mismatched: val1 != val2')