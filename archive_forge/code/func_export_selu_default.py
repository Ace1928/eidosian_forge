import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_selu_default() -> None:
    default_alpha = 1.6732631921768188
    default_gamma = 1.0507010221481323
    node = onnx.helper.make_node('Selu', inputs=['x'], outputs=['y'])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) * default_gamma + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
    expect(node, inputs=[x], outputs=[y], name='test_selu_default')