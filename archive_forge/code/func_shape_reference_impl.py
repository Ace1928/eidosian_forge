import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def shape_reference_impl(x, start=None, end=None):
    dims = x.shape[start:end]
    return np.array(dims).astype(np.int64)