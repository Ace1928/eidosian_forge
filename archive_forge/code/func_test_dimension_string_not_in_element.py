from holoviews.core import Dimension, Element
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_string_not_in_element(self):
    self.assertFalse('D' in self.element)