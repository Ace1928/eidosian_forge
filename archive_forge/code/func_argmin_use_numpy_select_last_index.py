import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def argmin_use_numpy_select_last_index(data: np.ndarray, axis: int=0, keepdims: int=True) -> np.ndarray:
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)