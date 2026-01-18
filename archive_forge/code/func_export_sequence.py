import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sequence() -> None:
    node = onnx.helper.make_node('Identity', inputs=['x'], outputs=['y'])
    data = [np.array([[[[1, 2], [3, 4]]]], dtype=np.float32), np.array([[[[2, 3], [1, 5]]]], dtype=np.float32)]
    expect(node, inputs=[data], outputs=[data], name='test_identity_sequence')