import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_axis() -> None:
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    normalized_shape = calculate_normalized_shape(X.shape, -1)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B)
    node = onnx.helper.make_node('LayerNormalization', inputs=['X', 'W', 'B'], outputs=['Y', 'Mean', 'InvStdDev'])
    expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name='test_layer_normalization_default_axis')