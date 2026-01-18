import datetime
import json
import numpy as np
from ase.utils import reader, writer
def create_ndarray(shape, dtype, data):
    """Create ndarray from shape, dtype and flattened data."""
    array = np.empty(shape, dtype=dtype)
    flatbuf = array.ravel()
    if np.iscomplexobj(array):
        flatbuf.dtype = array.real.dtype
    flatbuf[:] = data
    return array