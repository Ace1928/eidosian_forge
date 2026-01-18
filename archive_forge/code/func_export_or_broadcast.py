import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_or_broadcast() -> None:
    node = onnx.helper.make_node('Or', inputs=['x', 'y'], outputs=['or'])
    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    y = (np.random.randn(5) > 0).astype(bool)
    z = np.logical_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_or_bcast3v1d')
    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    y = (np.random.randn(4, 5) > 0).astype(bool)
    z = np.logical_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_or_bcast3v2d')
    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    y = (np.random.randn(5, 6) > 0).astype(bool)
    z = np.logical_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_or_bcast4v2d')
    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    y = (np.random.randn(4, 5, 6) > 0).astype(bool)
    z = np.logical_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_or_bcast4v3d')
    x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
    y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
    z = np.logical_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_or_bcast4v4d')