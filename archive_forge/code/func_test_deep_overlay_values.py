from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_deep_overlay_values(self):
    o1 = self.el1 * self.el2
    o2 = self.el1 * self.el2
    o3 = self.el1 * self.el2
    t = o1 * o2 * o3
    self.assertEqual(t.values(), [self.el1, self.el2, self.el1, self.el2, self.el1, self.el2])