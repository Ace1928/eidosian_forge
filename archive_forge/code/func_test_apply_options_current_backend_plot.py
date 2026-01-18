import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_current_backend_plot(self):
    obj = ExampleElement([]).options(plot_opt1='A')
    opts = Store.lookup_options('backend_1', obj, 'plot')
    assert opts.options == {'plot_opt1': 'A'}