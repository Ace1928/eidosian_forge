import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def calculate_normalized_shape(X_shape, axis):
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]