import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sub_broadcast() -> None:
    node = onnx.helper.make_node('Sub', inputs=['x', 'y'], outputs=['z'])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x - y
    expect(node, inputs=[x, y], outputs=[z], name='test_sub_bcast')