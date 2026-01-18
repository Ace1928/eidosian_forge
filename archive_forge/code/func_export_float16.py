import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_float16() -> None:
    node = onnx.helper.make_node('IsNaN', inputs=['x'], outputs=['y'])
    x = np.array([-1.2, np.nan, np.inf, 2.8, -np.inf, np.inf], dtype=np.float16)
    y = np.isnan(x)
    expect(node, inputs=[x], outputs=[y], name='test_isnan_float16')