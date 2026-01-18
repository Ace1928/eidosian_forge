import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_explicit_backend_plot_multiple(self):
    obj = ExampleElement([]).options(plot_opt1='A', plot_opt2='B', backend='backend_2')
    opts = Store.lookup_options('backend_2', obj, 'plot')
    assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}