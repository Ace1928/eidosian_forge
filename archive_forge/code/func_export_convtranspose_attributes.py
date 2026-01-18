import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_convtranspose_attributes() -> None:
    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]).astype(np.float32)
    W = np.array([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]).astype(np.float32)
    y = np.array([[[[0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]).astype(np.float32)
    node = onnx.helper.make_node('ConvTranspose', ['X', 'W'], ['Y'], strides=[3, 2], output_shape=[10, 8])
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_output_shape')
    node = onnx.helper.make_node('ConvTranspose', ['X', 'W'], ['Y'], strides=[3, 2], output_padding=[1, 1])
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pad')
    node = onnx.helper.make_node('ConvTranspose', ['X', 'W'], ['Y'], name='test', strides=[3, 2], output_shape=[10, 8], kernel_shape=[3, 3], output_padding=[1, 1])
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_kernel_shape')