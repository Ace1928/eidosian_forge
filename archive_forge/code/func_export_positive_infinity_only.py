import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_positive_infinity_only() -> None:
    node = onnx.helper.make_node('IsInf', inputs=['x'], outputs=['y'], detect_negative=0)
    x = np.array([-1.7, np.nan, np.inf, 3.6, -np.inf, np.inf], dtype=np.float32)
    y = np.isposinf(x)
    expect(node, inputs=[x], outputs=[y], name='test_isinf_positive')