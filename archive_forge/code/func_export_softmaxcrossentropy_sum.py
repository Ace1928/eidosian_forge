import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_sum() -> None:
    reduction = 'sum'
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y'], outputs=['z'], reduction=reduction)
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
    sce = softmaxcrossentropy(x, labels, reduction='sum')
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_sum')