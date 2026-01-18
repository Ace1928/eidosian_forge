import param
from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, Stream
def test_element_dynamic_with_instance_param(self):
    curve = Curve([1, 2, 3])
    inst = ParamClass(label='Test')
    applied = ExampleOperation(curve, label=inst.param.label)
    self.assertEqual(len(applied.streams), 1)
    self.assertIsInstance(applied.streams[0], Params)
    self.assertEqual(applied.streams[0].parameters, [inst.param.label])
    self.assertEqual(applied[()], curve.relabel('Test'))