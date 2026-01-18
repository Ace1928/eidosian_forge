from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_element_options_wrapping(self):
    element = ExampleElement(None).opts(plot_opt1='A' * 40, style_opt1='B' * 40, backend='backend_1')
    r = self.pprinter.pprint(element)
    self.assertEqual(r, ":ExampleElement\n | Options(plot_opt1='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',\n |         style_opt1='BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')")