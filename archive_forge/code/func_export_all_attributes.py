from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_all_attributes() -> None:
    node = onnx.helper.make_node('Gemm', inputs=['a', 'b', 'c'], outputs=['y'], alpha=0.25, beta=0.35, transA=1, transB=1)
    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
    expect(node, inputs=[a, b, c], outputs=[y], name='test_gemm_all_attributes')