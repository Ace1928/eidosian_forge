import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_without_output_shape() -> None:
    node = onnx.helper.make_node('MaxUnpool', inputs=['xT', 'xI'], outputs=['y'], kernel_shape=[2, 2], strides=[2, 2])
    xT = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    xI = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
    y = np.array([[[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]], dtype=np.float32)
    expect(node, inputs=[xT, xI], outputs=[y], name='test_maxunpool_export_without_output_shape')