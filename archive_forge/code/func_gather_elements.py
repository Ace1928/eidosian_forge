import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def gather_elements(data, indices, axis=0):
    data_swaped = np.swapaxes(data, 0, axis)
    index_swaped = np.swapaxes(indices, 0, axis)
    gathered = np.choose(index_swaped, data_swaped, mode='wrap')
    y = np.swapaxes(gathered, 0, axis)
    return y