import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_not_sorted_without_axis() -> None:
    node_not_sorted = onnx.helper.make_node('Unique', inputs=['X'], outputs=['Y', 'indices', 'inverse_indices', 'counts'], sorted=0)
    x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True)
    argsorted_indices = np.argsort(indices)
    inverse_indices_map = dict(zip(argsorted_indices, np.arange(len(argsorted_indices))))
    indices = indices[argsorted_indices]
    y = np.take(x, indices, axis=0)
    inverse_indices = np.asarray([inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
    counts = counts[argsorted_indices]
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    expect(node_not_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_not_sorted_without_axis')