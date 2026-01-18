from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_cyclic_unequal(self):
    try:
        self.assertEqual(self.dimension4, self.dimension5)
    except AssertionError as e:
        self.assertEqual(str(e), "Dimension parameter 'cyclic' mismatched: False != True")