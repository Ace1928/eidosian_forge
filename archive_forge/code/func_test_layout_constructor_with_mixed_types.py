from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_constructor_with_mixed_types(self):
    layout1 = self.el1 + self.el4 + self.el7
    layout2 = self.el2 + self.el5 + self.el8
    paths = Layout([layout1, layout2, self.el3]).keys()
    self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('ValA', 'LabelA'), ('Element', 'II'), ('ValB', 'I'), ('ValA', 'LabelB'), ('Element', 'III')])