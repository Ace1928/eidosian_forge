import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_dim_unchanged() -> None:
    node = onnx.helper.make_node('Expand', inputs=['data', 'new_shape'], outputs=['expanded'])
    shape = [3, 1]
    new_shape = [3, 4]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    expanded = np.tile(data, 4)
    new_shape = np.array(new_shape, dtype=np.int64)
    expect(node, inputs=[data, new_shape], outputs=[expanded], name='test_expand_dim_unchanged')