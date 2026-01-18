import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_explicit_backend_style_multiple(self):
    obj = ExampleElement([]).options(style_opt1='A', style_opt2='B', backend='backend_2')
    opts = Store.lookup_options('backend_2', obj, 'style')
    assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}