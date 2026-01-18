from holoviews.core import Dimension, Element
from holoviews.element.comparison import ComparisonTestCase
def test_key_dimension_in_element(self):
    self.assertTrue(Dimension('A') in self.element)