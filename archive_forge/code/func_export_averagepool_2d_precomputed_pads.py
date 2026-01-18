import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_averagepool_2d_precomputed_pads() -> None:
    """input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
    node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'], kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    x = np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]).astype(np.float32)
    y = np.array([[[[7, 7.5, 8, 8.5, 9], [9.5, 10, 10.5, 11, 11.5], [12, 12.5, 13, 13.5, 14], [14.5, 15, 15.5, 16, 16.5], [17, 17.5, 18, 18.5, 19]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_pads')