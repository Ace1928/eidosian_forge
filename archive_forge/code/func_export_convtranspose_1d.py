import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_convtranspose_1d() -> None:
    x = np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)
    W = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(np.float32)
    node = onnx.helper.make_node('ConvTranspose', ['X', 'W'], ['Y'])
    y = np.array([[[0.0, 1.0, 3.0, 3.0, 2.0], [0.0, 1.0, 3.0, 3.0, 2.0]]]).astype(np.float32)
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_1d')