import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_scatter_elements_without_axis() -> None:
    node = onnx.helper.make_node('ScatterElements', inputs=['data', 'indices', 'updates'], outputs=['y'])
    data = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
    updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)
    y = scatter_elements(data, indices, updates)
    expect(node, inputs=[data, indices, updates], outputs=[y], name='test_scatter_elements_without_axis')