from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def gemm_reference_implementation(A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray]=None, alpha: float=1.0, beta: float=1.0, transA: int=0, transB: int=0) -> np.ndarray:
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)
    Y = alpha * np.dot(A, B) + beta * C
    return Y