import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_negative_axes_keepdims() -> None:
    shape = [3, 2, 2]
    axes = np.array([-1], dtype=np.int64)
    keepdims = 1
    node = onnx.helper.make_node('ReduceL2', inputs=['data', 'axes'], outputs=['reduced'], keepdims=keepdims)
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    reduced = np.sqrt(np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_l2_negative_axes_keep_dims_example')
    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_l2_negative_axes_keep_dims_random')