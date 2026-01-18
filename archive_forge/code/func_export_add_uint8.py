import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_add_uint8() -> None:
    node = onnx.helper.make_node('Add', inputs=['x', 'y'], outputs=['sum'])
    x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    expect(node, inputs=[x, y], outputs=[x + y], name='test_add_uint8')