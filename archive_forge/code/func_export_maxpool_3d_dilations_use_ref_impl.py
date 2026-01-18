import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_3d_dilations_use_ref_impl() -> None:
    """input_shape: [1, 1, 4, 4, 4]
        output_shape: [1, 1, 2, 2, 2]
        """
    dilations = [2, 2, 2]
    kernel_shape = [2, 2, 2]
    strides = [1, 1, 1]
    ceil_mode = False
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=[2, 2, 2], strides=[1, 1, 1], dilations=dilations)
    x = np.array([[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]]).astype(np.float32)
    x_shape = x.shape[2:]
    out_shape, pads = get_output_shape_explicit_padding(None, x_shape, kernel_shape, strides, dilations, ceil_mode=ceil_mode)
    padded = x
    y = pool(padded, (1, 1, *x_shape), kernel_shape, strides, out_shape, 'MAX', pads, dilations=dilations)
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_dilations_use_ref_impl')