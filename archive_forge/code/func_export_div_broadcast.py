import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_div_broadcast() -> None:
    node = onnx.helper.make_node('Div', inputs=['x', 'y'], outputs=['z'])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32) + 1.0
    z = x / y
    expect(node, inputs=[x, y], outputs=[z], name='test_div_bcast')