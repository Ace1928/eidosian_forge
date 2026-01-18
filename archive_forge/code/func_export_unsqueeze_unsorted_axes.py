import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_unsqueeze_unsorted_axes() -> None:
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)
    node = onnx.helper.make_node('Unsqueeze', inputs=['x', 'axes'], outputs=['y'])
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)
    expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_unsorted_axes')