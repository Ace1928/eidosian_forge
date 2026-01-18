import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_hardmax_axis() -> None:
    x = np.random.randn(3, 4, 5).astype(np.float32)
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=0)
    y = hardmax(x, axis=0)
    expect(node, inputs=[x], outputs=[y], name='test_hardmax_axis_0')
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=1)
    y = hardmax(x, axis=1)
    expect(node, inputs=[x], outputs=[y], name='test_hardmax_axis_1')
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=2)
    y = hardmax(x, axis=2)
    expect(node, inputs=[x], outputs=[y], name='test_hardmax_axis_2')
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'], axis=-1)
    y = hardmax(x, axis=-1)
    expect(node, inputs=[x], outputs=[y], name='test_hardmax_negative_axis')
    node = onnx.helper.make_node('Hardmax', inputs=['x'], outputs=['y'])
    expect(node, inputs=[x], outputs=[y], name='test_hardmax_default_axis')