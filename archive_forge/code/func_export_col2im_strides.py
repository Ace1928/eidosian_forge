import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_col2im_strides() -> None:
    input = np.array([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]).astype(np.float32)
    image_shape = np.array([5, 5]).astype(np.int64)
    block_shape = np.array([3, 3]).astype(np.int64)
    output = np.array([[[[0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 2.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]]]]).astype(np.float32)
    node = onnx.helper.make_node('Col2Im', ['input', 'image_shape', 'block_shape'], ['output'], strides=[2, 2])
    expect(node, inputs=[input, image_shape, block_shape], outputs=[output], name='test_col2im_strides')