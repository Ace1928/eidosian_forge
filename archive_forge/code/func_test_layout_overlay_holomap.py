from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_overlay_holomap(self):
    t = (self.el1 + self.el2) * HoloMap({0: self.el3})
    self.assertEqual(t, Layout([HoloMap({0: self.el1 * self.el3}), HoloMap({0: self.el2 * self.el3})]))