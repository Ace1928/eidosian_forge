import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_empty_axes_input_noop() -> None:
    shape = [3, 2, 2]
    keepdims = 1
    node = onnx.helper.make_node('ReduceSum', inputs=['data', 'axes'], outputs=['reduced'], keepdims=keepdims, noop_with_empty_axes=True)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    axes = np.array([], dtype=np.int64)
    reduced = np.array(data)
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_empty_axes_input_noop_example')
    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.array(data)
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_negative_axes_keepdims_random')