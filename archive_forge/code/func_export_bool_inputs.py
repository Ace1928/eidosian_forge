import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_bool_inputs() -> None:
    axes = np.array([1], dtype=np.int64)
    keepdims = 1
    node = onnx.helper.make_node('ReduceMin', inputs=['data', 'axes'], outputs=['reduced'], keepdims=keepdims)
    data = np.array([[True, True], [True, False], [False, True], [False, False]])
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=bool(keepdims))
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_min_bool_inputs')