import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_float32() -> None:
    node = onnx.helper.make_node('GatherND', inputs=['data', 'indices'], outputs=['output'])
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 0)
    expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
    assert np.array_equal(output, expected_output)
    expect(node, inputs=[data, indices], outputs=[output], name='test_gathernd_example_float32')