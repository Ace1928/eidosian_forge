import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_compress_default_axis() -> None:
    node = onnx.helper.make_node('Compress', inputs=['input', 'condition'], outputs=['output'])
    input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    condition = np.array([0, 1, 0, 0, 1])
    output = np.compress(condition, input)
    expect(node, inputs=[input, condition.astype(bool)], outputs=[output], name='test_compress_default_axis')