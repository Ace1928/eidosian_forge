import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_cumsum_1d() -> None:
    node = onnx.helper.make_node('CumSum', inputs=['x', 'axis'], outputs=['y'])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
    axis = np.int32(0)
    y = np.array([1.0, 3.0, 6.0, 10.0, 15.0]).astype(np.float64)
    expect(node, inputs=[x, axis], outputs=[y], name='test_cumsum_1d')