from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_units_unequal(self):
    try:
        self.assertEqual(self.dimension6, self.dimension7)
    except AssertionError as e:
        self.assertEqual(str(e), "Dimension parameter 'unit' mismatched: None != 'ms'")