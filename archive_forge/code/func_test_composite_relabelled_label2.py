from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_composite_relabelled_label2(self):
    t = (self.el1 * self.el2).relabel(label='Label1') + (self.el1 * self.el2).relabel(group='Val1', label='Label2')
    self.assertEqual(t.keys(), [('Overlay', 'Label1'), ('Val1', 'Label2')])