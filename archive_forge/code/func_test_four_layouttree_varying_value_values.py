from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_four_layouttree_varying_value_values(self):
    t = self.el1 + self.el4 + self.el2 + self.el3
    self.assertEqual(t.values(), [self.el1, self.el4, self.el2, self.el3])