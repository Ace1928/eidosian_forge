from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimensioned_comparison_equal(self):
    """Note that the data is not compared at the Dimensioned level"""
    self.assertEqual(self.dimensioned1, Dimensioned('other_data', vdims=self.value_list1, kdims=self.key_list1))