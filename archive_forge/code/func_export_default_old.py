import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_old() -> None:
    node = onnx.helper.make_node('Dropout', inputs=['x'], outputs=['y'])
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = x
    expect(node, inputs=[x], outputs=[y], name='test_dropout_default_old', opset_imports=[helper.make_opsetid('', 11)])