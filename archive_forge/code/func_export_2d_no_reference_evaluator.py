import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_affine_grid import (
@staticmethod
def export_2d_no_reference_evaluator() -> None:
    theta_2d = create_theta_2d()
    N, C, H, W = (len(theta_2d), 3, 5, 6)
    data_size = (H, W)
    for align_corners in (0, 1):
        node = onnx.helper.make_node('AffineGrid', inputs=['theta', 'size'], outputs=['grid'], align_corners=align_corners)
        original_grid = construct_original_grid(data_size, align_corners)
        grid = apply_affine_transform(theta_2d, original_grid)
        test_name = 'test_affine_grid_2d'
        if align_corners == 1:
            test_name += '_align_corners'
        expect(node, inputs=[theta_2d, np.array([N, C, H, W], dtype=np.int64)], outputs=[grid], name=test_name)