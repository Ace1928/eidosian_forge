import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_deep_setitem_str(self):
    self.tree['C.I'] = 3
    self.assertEqual(self.tree.C.I, 3)