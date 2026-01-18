import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_2d() -> None:
    node = onnx.helper.make_node('Det', inputs=['x'], outputs=['y'])
    x = np.arange(4).reshape(2, 2).astype(np.float32)
    y = np.linalg.det(x)
    expect(node, inputs=[x], outputs=[y], name='test_det_2d')