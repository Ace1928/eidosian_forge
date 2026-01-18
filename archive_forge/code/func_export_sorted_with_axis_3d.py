import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sorted_with_axis_3d() -> None:
    node_sorted = onnx.helper.make_node('Unique', inputs=['X'], outputs=['Y', 'indices', 'inverse_indices', 'counts'], sorted=1, axis=1)
    x = np.array([[[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=1)
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis_3d')