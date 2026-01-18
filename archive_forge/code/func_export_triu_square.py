import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_triu_square() -> None:
    node = onnx.helper.make_node('Trilu', inputs=['x'], outputs=['y'])
    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    y = triu_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_triu_square')