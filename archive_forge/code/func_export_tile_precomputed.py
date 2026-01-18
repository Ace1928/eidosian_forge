import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_tile_precomputed() -> None:
    node = onnx.helper.make_node('Tile', inputs=['x', 'y'], outputs=['z'])
    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    repeats = np.array([2, 2], dtype=np.int64)
    z = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=np.float32)
    expect(node, inputs=[x, repeats], outputs=[z], name='test_tile_precomputed')