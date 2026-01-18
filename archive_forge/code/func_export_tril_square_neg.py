import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_tril_square_neg() -> None:
    node = onnx.helper.make_node('Trilu', inputs=['x', 'k'], outputs=['y'], upper=0)
    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    k = np.array(-1).astype(np.int64)
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_square_neg')