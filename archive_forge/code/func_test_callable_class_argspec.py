from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_callable_class_argspec(self):
    self.assertEqual(Callable(CallableClass()).argspec.args, [])
    self.assertEqual(Callable(CallableClass()).argspec.keywords, None)
    self.assertEqual(Callable(CallableClass()).argspec.varargs, 'testargs')