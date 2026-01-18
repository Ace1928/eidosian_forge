import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_equal_string_broadcast() -> None:
    node = onnx.helper.make_node('Equal', inputs=['x', 'y'], outputs=['z'])
    x = np.array(['string1', 'string2'], dtype=np.dtype(object))
    y = np.array(['string1'], dtype=np.dtype(object))
    z = np.equal(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_equal_string_broadcast')