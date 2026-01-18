import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_axis() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'])
    x = np.array([[[[-162, 10], [-100, 232], [-20, -50]], [[-76, 0], [0, 252], [32, -44]], [[245, -485], [-960, -270], [-375, -470]]]], dtype=np.float32)
    y_scale = np.array([2, 4, 5], dtype=np.float32)
    y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(np.uint8)
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_axis')