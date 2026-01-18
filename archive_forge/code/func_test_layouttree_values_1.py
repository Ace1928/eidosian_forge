from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_values_1(self):
    t = self.el1 + self.el2
    self.assertEqual(t.values(), [self.el1, self.el2])