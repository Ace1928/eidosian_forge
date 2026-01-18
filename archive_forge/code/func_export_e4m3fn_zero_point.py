import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_e4m3fn_zero_point() -> None:
    node = onnx.helper.make_node('DequantizeLinear', inputs=['x', 'x_scale', 'zero_point'], outputs=['y'], axis=0)
    x = make_tensor('x', TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
    zero_point = make_tensor('zero_point', TensorProto.FLOAT8E4M3FN, [1], [0])
    x_scale = np.float32(2)
    y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float32)
    expect(node, inputs=[x, x_scale, zero_point], outputs=[y], name='test_dequantizelinear_e4m3fn_zero_point')