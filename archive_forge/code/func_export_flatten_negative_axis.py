import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_flatten_negative_axis() -> None:
    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(np.float32)
    for i in range(-len(shape), 0):
        node = onnx.helper.make_node('Flatten', inputs=['a'], outputs=['b'], axis=i)
        new_shape = (np.prod(shape[0:i]).astype(int), -1)
        b = np.reshape(a, new_shape)
        expect(node, inputs=[a], outputs=[b], name='test_flatten_negative_axis' + str(abs(i)))