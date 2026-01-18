from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_kwargs_invocation(self):
    c = Callable(lambda x, y: x + y)
    c(x=1, y=4)
    self.assertEqual(c.kwargs, dict(x=1, y=4))