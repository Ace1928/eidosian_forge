import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default() -> None:
    seed = np.int64(0)
    node = onnx.helper.make_node('Dropout', inputs=['x'], outputs=['y'], seed=seed)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = dropout(x)
    expect(node, inputs=[x], outputs=[y], name='test_dropout_default')