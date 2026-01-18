from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_kdims_args_only(self):

    def fn(*args):
        A, B = args
        return Scatter([(B, 2)], label=A)
    dmap = DynamicMap(fn, kdims=['A', 'B'])
    self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))