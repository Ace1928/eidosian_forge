import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_mean_weights_ii() -> None:
    reduction = 'mean'
    ignore_index = np.int64(0)
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z'], reduction=reduction, ignore_index=ignore_index)
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
    labels[0] = np.int64(0)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)
    sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii')