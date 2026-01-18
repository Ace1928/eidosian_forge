import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_garbage_collect_cleans_unused_tree(self):
    obj = ExampleElement([]).opts(style_opt1='A')
    del obj
    gc.collect()
    custom_options = Store._custom_options['backend_1']
    self.assertEqual(len(custom_options), 0)