import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_mean_3d() -> None:
    reduction = 'mean'
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y'], outputs=['z'], reduction=reduction)
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    sce = softmaxcrossentropy(x, y)
    expect(node, inputs=[x, y], outputs=[sce], name='test_sce_mean_3d')