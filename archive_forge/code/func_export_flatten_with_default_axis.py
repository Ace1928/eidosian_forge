import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_flatten_with_default_axis() -> None:
    node = onnx.helper.make_node('Flatten', inputs=['a'], outputs=['b'])
    shape = (5, 4, 3, 2)
    a = np.random.random_sample(shape).astype(np.float32)
    new_shape = (5, 24)
    b = np.reshape(a, new_shape)
    expect(node, inputs=[a], outputs=[b], name='test_flatten_default_axis')