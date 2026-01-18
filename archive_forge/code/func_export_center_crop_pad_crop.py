import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_center_crop_pad_crop() -> None:
    node = onnx.helper.make_node('CenterCropPad', inputs=['x', 'shape'], outputs=['y'])
    x = np.random.randn(20, 10, 3).astype(np.float32)
    shape = np.array([10, 7, 3], dtype=np.int64)
    y = x[5:15, 1:8, :]
    expect(node, inputs=[x, shape], outputs=[y], name='test_center_crop_pad_crop')