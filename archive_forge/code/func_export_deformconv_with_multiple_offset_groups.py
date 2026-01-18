import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_deformconv_with_multiple_offset_groups() -> None:
    X = np.zeros((1, 2, 3, 3), dtype=np.float32)
    X[0, 0] = np.reshape(np.arange(9).astype(np.float32), (3, 3))
    X[0, 1] = np.reshape(np.arange(8, -1, -1).astype(np.float32), (3, 3))
    X.shape = (1, 2, 3, 3)
    W = np.ones((1, 2, 2, 2), dtype=np.float32)
    offset = np.zeros((1, 16, 2, 2), dtype=np.float32)
    offset[0, 0, 0, 0] = 0.5
    offset[0, 13, 0, 1] = -0.1
    node = onnx.helper.make_node('DeformConv', inputs=['X', 'W', 'offset'], outputs=['Y'], kernel_shape=[2, 2], pads=[0, 0, 0, 0], offset_group=2)
    Y = np.array([[[[33.5, 32.1], [32.0, 32.0]]]]).astype(np.float32)
    expect(node, inputs=[X, W, offset], outputs=[Y], name='test_deform_conv_with_multiple_offset_groups')