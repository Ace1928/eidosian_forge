import param
from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, Stream
def test_element_not_dynamic_with_param_method(self):
    curve = Curve([1, 2, 3])
    inst = ParamClass(label='Test')
    applied = ExampleOperation(curve, dynamic=False, label=inst.dynamic_label)
    self.assertEqual(applied, curve.relabel('Test!'))