import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_axes_keepdims_select_last_index() -> None:
    data = np.array([[2, 2], [3, 10]], dtype=np.float32)
    keepdims = 1
    node = onnx.helper.make_node('ArgMin', inputs=['data'], outputs=['result'], keepdims=keepdims, select_last_index=True)
    result = argmin_use_numpy_select_last_index(data, keepdims=keepdims)
    expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_example_select_last_index')
    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    result = argmin_use_numpy_select_last_index(data, keepdims=keepdims)
    expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_random_select_last_index')