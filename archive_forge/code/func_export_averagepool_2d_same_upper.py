import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_averagepool_2d_same_upper() -> None:
    """input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
        """
    node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'], kernel_shape=[2, 2], auto_pad='SAME_UPPER')
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_output_shape_auto_pad('SAME_UPPER', x_shape[2:], kernel_shape, strides)
    pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
    pad_top = pad_shape[0] // 2
    pad_bottom = pad_shape[0] - pad_top
    pad_left = pad_shape[1] // 2
    pad_right = pad_shape[1] - pad_left
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=np.nan)
    pads = (pad_top, pad_left, pad_bottom, pad_right)
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, 'AVG', pads)
    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_upper')