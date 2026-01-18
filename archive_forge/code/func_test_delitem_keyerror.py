import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_delitem_keyerror(self):
    with self.assertRaises(KeyError):
        del self.tree['C']