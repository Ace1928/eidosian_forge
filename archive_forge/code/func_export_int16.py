import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_int16() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'])
    x = np.array([0.0, -514.0, 3.0, -3.0, 2.9, -2.9, 3.1, -3.1, 65022.0, -66046.0, 65023.0, -66047.0, 65024.0, -66048.0, 70000.0, -70000.0]).astype(np.float32)
    y_scale = np.float32(2.0)
    y_zero_point = np.int16(256)
    y = np.array([256, -1, 258, 254, 257, 255, 258, 254, 32767, -32767, 32767, -32768, 32767, -32768, 32767, -32768]).astype(np.int16)
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_int16')