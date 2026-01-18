import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_float_ones() -> None:
    x = np.array([4, 3, 2]).astype(np.int64)
    tensor_value = onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1], [1])
    node = onnx.helper.make_node('ConstantOfShape', inputs=['x'], outputs=['y'], value=tensor_value)
    y = np.ones(x, dtype=np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_constantofshape_float_ones')