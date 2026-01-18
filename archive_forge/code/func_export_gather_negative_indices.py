import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gather_negative_indices() -> None:
    node = onnx.helper.make_node('Gather', inputs=['data', 'indices'], outputs=['y'], axis=0)
    data = np.arange(10).astype(np.float32)
    indices = np.array([0, -9, -10])
    y = np.take(data, indices, axis=0)
    expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y], name='test_gather_negative_indices')