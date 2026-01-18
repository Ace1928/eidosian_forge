import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_with_split_2() -> None:
    data = np.arange(18).reshape((3, 6)).astype(np.float32)
    split = np.array([1, 2], dtype=np.int64)
    node = onnx.helper.make_node('SplitToSequence', ['data', 'split'], ['seq'], axis=0)
    expected_outputs = [[data[:1], data[1:]]]
    expect(node, inputs=[data, split], outputs=expected_outputs, name='test_split_to_sequence_2')