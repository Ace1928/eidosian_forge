import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_col2im_pads() -> None:
    input = np.array([[[1.0, 6.0, 11.0, 16.0, 21.0, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71], [2.0, 7.0, 12.0, 17.0, 22.0, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72], [3.0, 8.0, 13.0, 18.0, 23.0, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73], [4.0, 9.0, 14.0, 19.0, 24.0, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74], [5.0, 10.0, 15.0, 20.0, 25.0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]]]).astype(np.float32)
    image_shape = np.array([5, 5]).astype(np.int64)
    block_shape = np.array([1, 5]).astype(np.int64)
    output = np.array([[[[8.0, 21.0, 24.0, 27.0, 24.0], [38.0, 66.0, 69.0, 72.0, 54.0], [68.0, 111.0, 114.0, 117.0, 84.0], [98.0, 156.0, 159.0, 162.0, 114.0], [128.0, 201.0, 204.0, 207.0, 144.0]]]]).astype(np.float32)
    node = onnx.helper.make_node('Col2Im', ['input', 'image_shape', 'block_shape'], ['output'], pads=[0, 1, 0, 1])
    expect(node, inputs=[input, image_shape, block_shape], outputs=[output], name='test_col2im_pads')