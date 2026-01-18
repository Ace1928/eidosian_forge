import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_lppool_2d_pads() -> None:
    """input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
    p = 3
    node = onnx.helper.make_node('LpPool', inputs=['x'], outputs=['y'], kernel_shape=[3, 3], pads=[2, 2, 2, 2], p=p)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = pad_top = pad_right = pad_left = 2
    pads = [pad_top, pad_left, pad_bottom, pad_right]
    out_shape, pads = get_output_shape_explicit_padding(pads, x_shape[2:], kernel_shape, strides)
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, 'LPPOOL', pads, p=p)
    expect(node, inputs=[x], outputs=[y], name='test_lppool_2d_pads')