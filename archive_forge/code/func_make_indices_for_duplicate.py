import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def make_indices_for_duplicate(idx):
    final_idx = []
    for i in range(len(idx[0])):
        final_idx.append(tuple((idx_element[i] for idx_element in idx)))
    return list(final_idx)