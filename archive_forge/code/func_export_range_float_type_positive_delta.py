import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_range_float_type_positive_delta() -> None:
    node = onnx.helper.make_node('Range', inputs=['start', 'limit', 'delta'], outputs=['output'])
    start = np.float32(1)
    limit = np.float32(5)
    delta = np.float32(2)
    output = np.arange(start, limit, delta, dtype=np.float32)
    expect(node, inputs=[start, limit, delta], outputs=[output], name='test_range_float_type_positive_delta')