import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_dim_changed() -> None:
    node = onnx.helper.make_node('Expand', inputs=['data', 'new_shape'], outputs=['expanded'])
    shape = [3, 1]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    new_shape = [2, 1, 6]
    expanded = data * np.ones(new_shape, dtype=np.float32)
    new_shape = np.array(new_shape, dtype=np.int64)
    expect(node, inputs=[data, new_shape], outputs=[expanded], name='test_expand_dim_changed')