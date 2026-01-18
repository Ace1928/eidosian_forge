import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_with_dtype() -> None:
    shape = (3, 4)
    node = onnx.helper.make_node('EyeLike', inputs=['x'], outputs=['y'], dtype=onnx.TensorProto.DOUBLE)
    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.float64)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_with_dtype')