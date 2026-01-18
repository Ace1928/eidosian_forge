import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_col2im_dilations() -> None:
    input = np.array([[[1.0, 5.0, 9.0, 13.0, 17], [2.0, 6.0, 10.0, 14.0, 18], [3.0, 7.0, 11.0, 15.0, 19], [4.0, 8.0, 12.0, 16.0, 20]]]).astype(np.float32)
    image_shape = np.array([6, 6]).astype(np.int64)
    block_shape = np.array([2, 2]).astype(np.int64)
    output = np.array([[[[1.0, 0.0, 0.0, 0.0, 0.0, 2.0], [8.0, 0.0, 0.0, 0.0, 0.0, 10.0], [16.0, 0.0, 0.0, 0.0, 0.0, 18.0], [24.0, 0.0, 0.0, 0.0, 0.0, 26.0], [32.0, 0.0, 0.0, 0.0, 0.0, 34.0], [19.0, 0.0, 0.0, 0.0, 0.0, 20.0]]]]).astype(np.float32)
    node = onnx.helper.make_node('Col2Im', ['input', 'image_shape', 'block_shape'], ['output'], dilations=[1, 5])
    expect(node, inputs=[input, image_shape, block_shape], outputs=[output], name='test_col2im_dilations')