from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_zero_bias() -> None:
    node = onnx.helper.make_node('Gemm', inputs=['a', 'b', 'c'], outputs=['y'])
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y], name='test_gemm_default_zero_bias')