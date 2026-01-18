import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_training_default() -> None:
    seed = np.int64(0)
    node = onnx.helper.make_node('Dropout', inputs=['x', 'r', 't'], outputs=['y'], seed=seed)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.5)
    t = np.bool_(True)
    y = dropout(x, r, training_mode=t)
    expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout_default')