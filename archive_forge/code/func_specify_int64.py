import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def specify_int64(indices, inverse_indices, counts):
    return (np.array(indices, dtype=np.int64), np.array(inverse_indices, dtype=np.int64), np.array(counts, dtype=np.int64))