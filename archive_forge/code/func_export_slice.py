import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_slice() -> None:
    node = onnx.helper.make_node('Slice', inputs=['x', 'starts', 'ends', 'axes', 'steps'], outputs=['y'])
    x = np.random.randn(20, 10, 5).astype(np.float32)
    y = x[0:3, 0:10]
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)
    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y], name='test_slice')