import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1_weight() -> None:
    reduction = 'mean'
    node = onnx.helper.make_node('NegativeLogLikelihoodLoss', inputs=['input', 'target', 'weight'], outputs=['loss'], reduction=reduction)
    N, C, d1 = (3, 5, 2)
    np.random.seed(0)
    input = np.random.rand(N, C, d1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)
    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)
    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss], name='test_nllloss_NCd1_weight')