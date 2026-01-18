import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_logsoftmax_axis() -> None:
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    y = logsoftmax(x)
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'])
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_large_number')
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=0)
    y = logsoftmax(x, axis=0)
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_axis_0')
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=1)
    y = logsoftmax(x, axis=1)
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_axis_1')
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=2)
    y = logsoftmax(x, axis=2)
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_axis_2')
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'], axis=-1)
    y = logsoftmax(x, axis=-1)
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_negative_axis')
    node = onnx.helper.make_node('LogSoftmax', inputs=['x'], outputs=['y'])
    expect(node, inputs=[x], outputs=[y], name='test_logsoftmax_default_axis')