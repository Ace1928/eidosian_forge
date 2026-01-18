import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.random.standard_normal', v1=[])
@np_utils.np_doc('random.standard_normal')
def standard_normal(size=None):
    if size is None:
        size = ()
    elif np_utils.isscalar(size):
        size = (size,)
    dtype = np_utils.result_type(float)
    return random_ops.random_normal(size, dtype=dtype)