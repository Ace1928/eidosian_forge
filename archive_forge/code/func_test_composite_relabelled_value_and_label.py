from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_composite_relabelled_value_and_label(self):
    t = (self.el1 * self.el2).relabel(group='Val1', label='Label1') + (self.el1 * self.el2).relabel(group='Val2', label='Label2')
    self.assertEqual(t.keys(), [('Val1', 'Label1'), ('Val2', 'Label2')])