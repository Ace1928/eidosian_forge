import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN
@staticmethod
def export_adagrad_multiple() -> None:
    norm_coefficient = 0.001
    epsilon = 1e-05
    decay_factor = 0.1
    node = onnx.helper.make_node('Adagrad', inputs=['R', 'T', 'X1', 'X2', 'G1', 'G2', 'H1', 'H2'], outputs=['X1_new', 'X2_new', 'H1_new', 'H2_new'], norm_coefficient=norm_coefficient, epsilon=epsilon, decay_factor=decay_factor, domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)
    r = np.array(0.1, dtype=np.float32)
    t = np.array(0, dtype=np.int64)
    x1 = np.array([1.0], dtype=np.float32)
    g1 = np.array([-1.0], dtype=np.float32)
    h1 = np.array([2.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0], dtype=np.float32)
    g2 = np.array([-1.0, -3.0], dtype=np.float32)
    h2 = np.array([4.0, 1.0], dtype=np.float32)
    x1_new, h1_new = apply_adagrad(r, t, x1, g1, h1, norm_coefficient, epsilon, decay_factor)
    x2_new, h2_new = apply_adagrad(r, t, x2, g2, h2, norm_coefficient, epsilon, decay_factor)
    expect(node, inputs=[r, t, x1, x2, g1, g2, h1, h2], outputs=[x1_new, x2_new, h1_new, h2_new], name='test_adagrad_multiple', opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])