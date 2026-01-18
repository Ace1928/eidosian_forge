import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_deformconv_with_mask_bias() -> None:
    X = np.arange(9).astype(np.float32)
    X.shape = (1, 1, 3, 3)
    W = np.ones((1, 1, 2, 2), dtype=np.float32)
    B = np.ones((1,), dtype=np.float32)
    offset = np.zeros((1, 8, 2, 2), dtype=np.float32)
    offset[0, 0, 0, 0] = 0.5
    offset[0, 5, 0, 1] = -0.1
    mask = np.ones((1, 4, 2, 2), dtype=np.float32)
    mask[0, 2, 1, 1] = 0.2
    node = onnx.helper.make_node('DeformConv', inputs=['X', 'W', 'offset', 'B', 'mask'], outputs=['Y'], kernel_shape=[2, 2], pads=[0, 0, 0, 0])
    Y = np.array([[[[10.5, 12.9], [21.0, 19.4]]]]).astype(np.float32)
    expect(node, inputs=[X, W, offset, B, mask], outputs=[Y], name='test_deform_conv_with_mask_bias')