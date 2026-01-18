from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimensioned_comparison_unequal_group(self):
    try:
        self.assertEqual(self.dimensioned1, self.dimensioned6)
    except AssertionError as e:
        self.assertEqual(str(e), 'Group labels mismatched.')