import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_scatternd_add() -> None:
    node = onnx.helper.make_node('ScatterND', inputs=['data', 'indices', 'updates'], outputs=['y'], reduction='add')
    data = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]], [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]], [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]], [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
    indices = np.array([[0], [0]], dtype=np.int64)
    updates = np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
    output = scatter_nd_impl(data, indices, updates, reduction='add')
    expect(node, inputs=[data, indices, updates], outputs=[output], name='test_scatternd_add')