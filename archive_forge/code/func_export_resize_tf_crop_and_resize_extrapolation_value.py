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
def export_resize_tf_crop_and_resize_extrapolation_value() -> None:
    node = onnx.helper.make_node('Resize', inputs=['X', 'roi', '', 'sizes'], outputs=['Y'], mode='linear', coordinate_transformation_mode='tf_crop_and_resize', extrapolation_value=10.0)
    data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    output = interpolate_nd(data, lambda x, _: linear_coeffs(x), output_size=sizes, roi=roi, coordinate_transformation_mode='tf_crop_and_resize', extrapolation_value=10.0).astype(np.float32)
    expect(node, inputs=[data, roi, sizes], outputs=[output], name='test_resize_tf_crop_and_resize')