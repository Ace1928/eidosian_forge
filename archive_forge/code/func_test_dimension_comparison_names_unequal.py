from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_names_unequal(self):
    try:
        self.assertEqual(self.dimension1, self.dimension2)
    except AssertionError as e:
        self.assertEqual(str(e), 'Dimension names mismatched: dim1 != dim2')