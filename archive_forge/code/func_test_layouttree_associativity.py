from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_associativity(self):
    t1 = self.el1 + self.el2 + self.el3
    t2 = self.el1 + self.el2 + self.el3
    t3 = self.el1 + (self.el2 + self.el3)
    self.assertEqual(t1.keys(), t2.keys())
    self.assertEqual(t2.keys(), t3.keys())