import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_populate_off_main_diagonal() -> None:
    shape = (4, 5)
    off_diagonal_offset = 1
    node = onnx.helper.make_node('EyeLike', inputs=['x'], outputs=['y'], k=off_diagonal_offset, dtype=onnx.TensorProto.FLOAT)
    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_populate_off_main_diagonal')