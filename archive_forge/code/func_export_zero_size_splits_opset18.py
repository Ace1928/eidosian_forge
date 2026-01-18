import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_zero_size_splits_opset18() -> None:
    node_input = np.array([]).astype(np.float32)
    split = np.array([0, 0, 0]).astype(np.int64)
    node = onnx.helper.make_node('Split', inputs=['input', 'split'], outputs=['output_1', 'output_2', 'output_3'])
    expected_outputs = [np.array([]).astype(np.float32), np.array([]).astype(np.float32), np.array([]).astype(np.float32)]
    expect(node, inputs=[node_input, split], outputs=expected_outputs, name='test_split_zero_size_splits_opset18')