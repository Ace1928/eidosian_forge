import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_cloned(self):
    obj1 = ExampleElement([])
    obj2 = obj1.options(style_opt1='A')
    opts = Store.lookup_options('backend_1', obj2, 'style')
    assert opts.options == {'style_opt1': 'A'}
    assert obj1 is not obj2