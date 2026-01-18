import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_mod_mixed_sign_float32() -> None:
    node = onnx.helper.make_node('Mod', inputs=['x', 'y'], outputs=['z'], fmod=1)
    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
    z = np.fmod(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_mod_mixed_sign_float32')