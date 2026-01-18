from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_callable_lambda_extras_kwargs(self):
    substr = "['x'] overridden by keywords"
    self.assertEqual(Callable(lambda x, y: x + y)(3, 5, x=10), 15)
    self.log_handler.assertEndsWith('WARNING', substr)