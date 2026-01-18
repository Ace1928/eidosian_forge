import builtins
import collections
import math
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import convert_to_tensor
def divide_no_nan(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(getattr(x1, 'dtype', type(x1)), getattr(x2, 'dtype', type(x2)), float)
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.math.divide_no_nan(x1, x2)