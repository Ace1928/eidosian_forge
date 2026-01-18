import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_non_reduced_axis_zero() -> None:
    """Test case with the non-reduced-axis of size zero."""
    shape = [2, 0, 4]
    keepdims = 1
    reduced_shape = [2, 0, 1]
    node = onnx.helper.make_node('ReduceSum', inputs=['data', 'axes'], outputs=['reduced'], keepdims=keepdims)
    data = np.array([], dtype=np.float32).reshape(shape)
    axes = np.array([2], dtype=np.int64)
    reduced = np.array([], dtype=np.float32).reshape(reduced_shape)
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_empty_set_non_reduced_axis_zero')