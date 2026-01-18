from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def output_result(self, B: int, frequencies: List[int]) -> np.ndarray:
    l_output_dims: List[int] = []
    if B == 0:
        l_output_dims.append(self.output_size_)
        B = 1
    else:
        l_output_dims.append(B)
        l_output_dims.append(self.output_size_)
    output_dims = tuple(l_output_dims)
    row_size = self.output_size_
    total_dims = np.prod(output_dims)
    Y = np.empty((total_dims,), dtype=np.float32)
    w = self.weights_
    if self.weighting_criteria_ == WeightingCriteria.TF:
        for i, f in enumerate(frequencies):
            Y[i] = f
    elif self.weighting_criteria_ == WeightingCriteria.IDF:
        if len(w) > 0:
            p = 0
            for _batch in range(B):
                for i in range(row_size):
                    Y[p] = w[i] if frequencies[p] > 0 else 0
                    p += 1
        else:
            p = 0
            for f in frequencies:
                Y[p] = 1 if f > 0 else 0
                p += 1
    elif self.weighting_criteria_ == WeightingCriteria.TFIDF:
        if len(w) > 0:
            p = 0
            for _batch in range(B):
                for i in range(row_size):
                    Y[p] = w[i] * frequencies[p]
                    p += 1
        else:
            p = 0
            for f in frequencies:
                Y[p] = f
                p += 1
    else:
        raise RuntimeError('Unexpected weighting_criteria.')
    return Y.reshape(output_dims)