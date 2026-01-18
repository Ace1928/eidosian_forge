from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_constructor2(self):
    t = Layout([self.el8])
    self.assertEqual(t.keys(), [('ValA', 'LabelB')])