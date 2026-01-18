import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_explicit_backend_persists_other_backend_inverted(self):
    obj = ExampleElement([])
    obj.opts(style_opt1='A', plot_opt1='B', backend='backend_2')
    obj.opts(style_opt1='C', plot_opt1='D', backend='backend_1')
    plot_opts = Store.lookup_options('backend_1', obj, 'plot')
    assert plot_opts.options == {'plot_opt1': 'D'}
    style_opts = Store.lookup_options('backend_1', obj, 'style')
    assert style_opts.options == {'style_opt1': 'C'}
    plot_opts = Store.lookup_options('backend_2', obj, 'plot')
    assert plot_opts.options == {'plot_opt1': 'B'}
    style_opts = Store.lookup_options('backend_2', obj, 'style')
    assert style_opts.options == {'style_opt1': 'A'}