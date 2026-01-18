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
def sparse_dense_matmul_3d(a, b):
    return tf.map_fn(lambda x: tf.sparse.sparse_dense_matmul(x[0], x[1]), elems=(a, b), fn_output_signature=a.dtype)