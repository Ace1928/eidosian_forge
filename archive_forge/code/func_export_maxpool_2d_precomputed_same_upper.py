import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_2d_precomputed_same_upper() -> None:
    """input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=[3, 3], strides=[2, 2], auto_pad='SAME_UPPER')
    x = np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]).astype(np.float32)
    y = np.array([[[[7, 9, 10], [17, 19, 20], [22, 24, 25]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_same_upper')