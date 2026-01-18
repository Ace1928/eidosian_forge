import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.random.poisson', v1=[])
@np_utils.np_doc('random.poisson')
def poisson(lam=1.0, size=None):
    if size is None:
        size = ()
    elif np_utils.isscalar(size):
        size = (size,)
    return random_ops.random_poisson(shape=size, lam=lam, dtype=np_dtypes.int_)