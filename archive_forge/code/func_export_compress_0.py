import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_compress_0() -> None:
    node = onnx.helper.make_node('Compress', inputs=['input', 'condition'], outputs=['output'], axis=0)
    input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1, 1])
    output = np.compress(condition, input, axis=0)
    expect(node, inputs=[input, condition.astype(bool)], outputs=[output], name='test_compress_0')