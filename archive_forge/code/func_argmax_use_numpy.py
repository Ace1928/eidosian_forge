import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def argmax_use_numpy(data: np.ndarray, axis: int=0, keepdims: int=1) -> np.ndarray:
    result = np.argmax(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)