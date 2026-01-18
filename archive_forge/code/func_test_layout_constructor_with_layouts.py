from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_constructor_with_layouts(self):
    layout1 = self.el1 + self.el4
    layout2 = self.el2 + self.el5
    paths = Layout([layout1, layout2]).keys()
    self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('ValB', 'I')])