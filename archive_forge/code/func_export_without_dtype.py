import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_without_dtype() -> None:
    shape = (4, 4)
    node = onnx.helper.make_node('EyeLike', inputs=['x'], outputs=['y'])
    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.int32)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_without_dtype')