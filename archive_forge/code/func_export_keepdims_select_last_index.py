import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_keepdims_select_last_index() -> None:
    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    axis = 1
    keepdims = 1
    node = onnx.helper.make_node('ArgMin', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims, select_last_index=True)
    result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
    expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_example_select_last_index')
    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
    expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_random_select_last_index')