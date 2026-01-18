import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gather_elements_negative_indices() -> None:
    axis = 0
    node = onnx.helper.make_node('GatherElements', inputs=['data', 'indices'], outputs=['y'], axis=axis)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[-1, -2, 0], [-2, 0, 0]], dtype=np.int32)
    y = gather_elements(data, indices, axis)
    expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y], name='test_gather_elements_negative_indices')