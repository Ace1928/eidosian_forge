from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_values_2(self):
    t = Layout([self.el1, self.el2])
    self.assertEqual(t.values(), [self.el1, self.el2])