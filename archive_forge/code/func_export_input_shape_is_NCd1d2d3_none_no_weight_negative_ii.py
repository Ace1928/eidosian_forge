import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1d2d3_none_no_weight_negative_ii() -> None:
    reduction = 'none'
    ignore_index = np.int64(-5)
    node = onnx.helper.make_node('NegativeLogLikelihoodLoss', inputs=['input', 'target'], outputs=['loss'], reduction=reduction, ignore_index=ignore_index)
    N, C, dim1, dim2, dim3 = (3, 5, 6, 6, 5)
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
    target[0][0][0][0] = -5
    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, reduction=reduction, ignore_index=ignore_index)
    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss], name='test_nllloss_NCd1d2d3_none_no_weight_negative_ii')