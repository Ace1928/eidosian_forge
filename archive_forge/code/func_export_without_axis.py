import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_without_axis() -> None:
    on_value = 5
    off_value = 2
    output_type = np.int32
    node = onnx.helper.make_node('OneHot', inputs=['indices', 'depth', 'values'], outputs=['y'])
    indices = np.array([0, 7, 8], dtype=np.int64)
    depth = np.float32(12)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_without_axis')