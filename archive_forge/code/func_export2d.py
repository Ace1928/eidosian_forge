import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export2d() -> None:
    X = np.random.randn(3, 4).astype(np.float32)

    def case(axis: int) -> None:
        normalized_shape = calculate_normalized_shape(X.shape, axis)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        B = np.random.randn(*normalized_shape).astype(np.float32)
        Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis=axis)
        node = onnx.helper.make_node('LayerNormalization', inputs=['X', 'W', 'B'], outputs=['Y', 'Mean', 'InvStdDev'], axis=axis)
        if axis < 0:
            name = f'test_layer_normalization_2d_axis_negative_{-axis}'
        else:
            name = f'test_layer_normalization_2d_axis{axis}'
        expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name=name)
    for i in range(len(X.shape)):
        case(i)
        case(i - len(X.shape))