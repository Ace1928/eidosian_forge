import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_int32_shape_zero() -> None:
    x = np.array([0]).astype(np.int64)
    tensor_value = onnx.helper.make_tensor('value', onnx.TensorProto.INT32, [1], [0])
    node = onnx.helper.make_node('ConstantOfShape', inputs=['x'], outputs=['y'], value=tensor_value)
    y = np.zeros(x, dtype=np.int32)
    expect(node, inputs=[x], outputs=[y], name='test_constantofshape_int_shape_zero')