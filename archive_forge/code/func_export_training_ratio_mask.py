import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_training_ratio_mask() -> None:
    seed = np.int64(0)
    node = onnx.helper.make_node('Dropout', inputs=['x', 'r', 't'], outputs=['y', 'z'], seed=seed)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.75)
    t = np.bool_(True)
    y, z = dropout(x, r, training_mode=t, return_mask=True)
    expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_mask')