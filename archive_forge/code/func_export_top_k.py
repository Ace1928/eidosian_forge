import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_top_k() -> None:
    axis = 1
    largest = 1
    k = 3
    node = onnx.helper.make_node('TopK', inputs=['x', 'k'], outputs=['values', 'indices'], axis=axis)
    X = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.float32)
    K = np.array([k], dtype=np.int64)
    values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)
    expect(node, inputs=[X, K], outputs=[values_ref, indices_ref], name='test_top_k')