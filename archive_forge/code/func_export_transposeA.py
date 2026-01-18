from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_transposeA() -> None:
    node = onnx.helper.make_node('Gemm', inputs=['a', 'b', 'c'], outputs=['y'], transA=1)
    a = np.random.ranf([6, 3]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transA=1)
    expect(node, inputs=[a, b, c], outputs=[y], name='test_gemm_transposeA')