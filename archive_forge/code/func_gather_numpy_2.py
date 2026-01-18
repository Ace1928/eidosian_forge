import numpy as np
from onnx.reference.op_run import OpRun
def gather_numpy_2(self: np.ndarray, index: np.ndarray) -> np.ndarray:
    res = []
    for a, b in zip(self, index):
        res.append(a[b[0]])
    return np.array(res, dtype=self.dtype).reshape(index.shape)