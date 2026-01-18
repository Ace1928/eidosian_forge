import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_mod_uint8() -> None:
    node = onnx.helper.make_node('Mod', inputs=['x', 'y'], outputs=['z'])
    x = np.array([4, 7, 5]).astype(np.uint8)
    y = np.array([2, 3, 8]).astype(np.uint8)
    z = np.mod(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_mod_uint8')