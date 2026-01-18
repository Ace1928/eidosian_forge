import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_mean_weights_log_prob() -> None:
    reduction = 'mean'
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z', 'log_prob'], reduction=reduction)
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, get_log_prob=True)
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_log_prob')