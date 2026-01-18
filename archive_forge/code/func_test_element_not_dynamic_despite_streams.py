import param
from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, Stream
def test_element_not_dynamic_despite_streams(self):
    curve = Curve([1, 2, 3])
    applied = Operation(curve, dynamic=False, streams=[Stream])
    self.assertEqual(applied, curve)