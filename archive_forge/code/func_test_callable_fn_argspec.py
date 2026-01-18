from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_callable_fn_argspec(self):

    def callback(x):
        return x
    self.assertEqual(Callable(callback).argspec.args, ['x'])
    self.assertEqual(Callable(callback).argspec.keywords, None)