import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_3d_dilations_use_ref_impl_large() -> None:
    x_shape = (32, 32, 32)
    dilations = (2, 2, 2)
    kernel_shape = (5, 5, 5)
    strides = (3, 3, 3)
    ceil_mode = True
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=kernel_shape, strides=strides, dilations=dilations, ceil_mode=ceil_mode)
    x = np.random.randn(1, 1, *x_shape).astype(np.float32)
    out_shape, pads = get_output_shape_explicit_padding(None, x_shape, kernel_shape, strides, dilations, ceil_mode=ceil_mode)
    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[3]), (pads[1], pads[4]), (pads[2], pads[5])), mode='constant', constant_values=0)
    y = pool(padded, (1, 1, *x_shape), kernel_shape, strides, out_shape, 'MAX', pads, dilations=dilations)
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_dilations_use_ref_impl_large')