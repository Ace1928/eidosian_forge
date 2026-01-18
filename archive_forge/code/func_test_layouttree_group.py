from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_group(self):
    t1 = self.el1 + self.el2
    t2 = Layout(list(t1.relabel(group='NewValue')))
    self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])