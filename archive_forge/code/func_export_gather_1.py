import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gather_1() -> None:
    node = onnx.helper.make_node('Gather', inputs=['data', 'indices'], outputs=['y'], axis=1)
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=1)
    expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y], name='test_gather_1')