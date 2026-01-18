from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_overlay_holomap_reverse(self):
    t = HoloMap({0: self.el3}) * (self.el1 + self.el2)
    self.assertEqual(t, Layout([HoloMap({0: self.el3 * self.el1}), HoloMap({0: self.el3 * self.el2})]))