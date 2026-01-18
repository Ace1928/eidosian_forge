import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_pow_broadcast() -> None:
    node = onnx.helper.make_node('Pow', inputs=['x', 'y'], outputs=['z'])
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array(2).astype(np.float32)
    z = pow(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_pow_bcast_scalar')
    node = onnx.helper.make_node('Pow', inputs=['x', 'y'], outputs=['z'])
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([1, 2, 3]).astype(np.float32)
    z = pow(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_pow_bcast_array')