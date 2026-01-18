import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_uint16() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'])
    x = np.array([0.0, -128.0, 3.0, -3.0, 2.9, -2.9, 3.1, -3.1, 65536.0, -65534.0, 70000.0, -70000.0]).astype(np.float32)
    y_scale = np.float32(2.0)
    y_zero_point = np.uint16(32767)
    y = np.array([32767, 32703, 32769, 32765, 32768, 32766, 32769, 32765, 65535, 0, 65535, 0]).astype(np.uint16)
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_uint16')