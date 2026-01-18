import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_mod_broadcast() -> None:
    node = onnx.helper.make_node('Mod', inputs=['x', 'y'], outputs=['z'])
    x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
    y = np.array([7]).astype(np.int32)
    z = np.mod(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_mod_broadcast')