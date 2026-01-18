import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_left_unit16() -> None:
    node = onnx.helper.make_node('BitShift', inputs=['x', 'y'], outputs=['z'], direction='LEFT')
    x = np.array([16, 4, 1]).astype(np.uint16)
    y = np.array([1, 2, 3]).astype(np.uint16)
    z = x << y
    expect(node, inputs=[x, y], outputs=[z], name='test_bitshift_left_uint16')