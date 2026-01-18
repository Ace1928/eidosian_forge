import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def preprocess_float(x):
    if is_prefer_float32():
        if isinstance(x, float):
            return np.float32(x)
        elif isinstance(x, complex):
            return np.complex64(x)
    return x