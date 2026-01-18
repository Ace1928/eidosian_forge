from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_split_kdims_and_streams(self):

    def fn(A, x=1, y=2):
        return Scatter([(x, y)], label=A)
    xy = streams.PointerXY(x=1, y=2)
    dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
    self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))