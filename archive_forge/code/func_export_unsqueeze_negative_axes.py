import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_unsqueeze_negative_axes() -> None:
    node = onnx.helper.make_node('Unsqueeze', inputs=['x', 'axes'], outputs=['y'])
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    y = np.expand_dims(x, axis=-2)
    expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_negative_axes')