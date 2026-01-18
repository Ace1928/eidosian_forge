import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_deep_delitem(self):
    BTree = self.tree.B
    del self.tree['B', 'II']
    self.assertIsInstance(self.tree.B.II, AttrTree)
    self.assertIs(self.tree.B, BTree)
    self.assertNotIn(('B', 'II'), self.tree.data)