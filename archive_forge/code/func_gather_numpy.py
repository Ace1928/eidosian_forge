import numpy as np
from onnx.reference.op_run import OpRun
def gather_numpy(self: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(f'Except for dimension {dim!r}, all dimensions of index and self should be the same size.')
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    try:
        gathered = np.choose(index_swaped, data_swaped, mode='wrap')
    except ValueError as e:
        if len(index_swaped.shape) == 2 and len(data_swaped.shape) == 2:
            return gather_numpy_2(self, index)
        raise e
    return np.swapaxes(gathered, 0, dim)