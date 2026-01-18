import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_empty_string_split() -> None:
    node = onnx.helper.make_node('StringSplit', inputs=['x'], outputs=['substrings', 'length'], delimiter=None, maxsplit=None)
    x = np.array([]).astype(object)
    substrings = np.array([]).astype(object).reshape(0, 0)
    length = np.array([], dtype=np.int64)
    expect(node, inputs=[x], outputs=[substrings, length], name='test_string_split_empty_tensor', output_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.STRING, (0, None)), None])