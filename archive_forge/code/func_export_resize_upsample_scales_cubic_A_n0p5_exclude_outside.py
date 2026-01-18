import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_resize import _cubic_coeffs as cubic_coeffs
from onnx.reference.ops.op_resize import (
from onnx.reference.ops.op_resize import _interpolate_nd as interpolate_nd
from onnx.reference.ops.op_resize import _linear_coeffs as linear_coeffs
from onnx.reference.ops.op_resize import (
from onnx.reference.ops.op_resize import _nearest_coeffs as nearest_coeffs
@staticmethod
def export_resize_upsample_scales_cubic_A_n0p5_exclude_outside() -> None:
    node = onnx.helper.make_node('Resize', inputs=['X', '', 'scales'], outputs=['Y'], mode='cubic', cubic_coeff_a=-0.5, exclude_outside=True)
    data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    output = interpolate_nd(data, lambda x, _: cubic_coeffs(x, A=-0.5), scale_factors=scales, exclude_outside=True).astype(np.float32)
    expect(node, inputs=[data, scales], outputs=[output], name='test_resize_upsample_scales_cubic_A_n0p5_exclude_outside')