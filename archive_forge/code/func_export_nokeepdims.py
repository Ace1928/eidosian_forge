import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_nokeepdims() -> None:
    data = np.arange(18).reshape((3, 6)).astype(np.float32)
    node = onnx.helper.make_node('SplitToSequence', ['data'], ['seq'], axis=1, keepdims=0)
    expected_outputs = [[data[:, i] for i in range(data.shape[1])]]
    expect(node, inputs=[data], outputs=expected_outputs, name='test_split_to_sequence_nokeepdims')