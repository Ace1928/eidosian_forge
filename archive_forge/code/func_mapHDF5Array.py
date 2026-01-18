import copy
import os
import pickle
import warnings
import numpy as np
@staticmethod
def mapHDF5Array(data, writable=False):
    off = data.id.get_offset()
    if writable:
        mode = 'r+'
    else:
        mode = 'r'
    if off is None:
        raise Exception('This dataset uses chunked storage; it can not be memory-mapped. (store using mappable=True)')
    return np.memmap(filename=data.file.filename, offset=off, dtype=data.dtype, shape=data.shape, mode=mode)