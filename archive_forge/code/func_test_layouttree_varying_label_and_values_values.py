from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_varying_label_and_values_values(self):
    t = self.el6 + self.el7 + self.el8
    self.assertEqual(t.values(), [self.el6, self.el7, self.el8])