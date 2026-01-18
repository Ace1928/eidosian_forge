from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_deep_composite_keys(self):
    o1 = self.el1 * self.el2
    o2 = self.el1 * self.el2
    o3 = self.el7 * self.el8
    t = o1 + o2 + o3
    expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
    self.assertEqual(t.keys(), expected_keys)