from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_kdims_only_invalid(self):

    def fn(A, B):
        return Scatter([(B, 2)], label=A)
    regexp = "Callable 'fn' accepts more positional arguments than there are kdims and stream parameters"
    with self.assertRaisesRegex(KeyError, regexp):
        DynamicMap(fn, kdims=['A'])