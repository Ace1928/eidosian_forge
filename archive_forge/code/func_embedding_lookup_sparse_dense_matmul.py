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
def embedding_lookup_sparse_dense_matmul(a, b):
    a, _ = tf.sparse.fill_empty_rows(a, 0)
    ids = tf.SparseTensor(indices=a.indices, values=a.indices[:, 1], dense_shape=a.dense_shape)
    return tf.nn.embedding_lookup_sparse(b, ids, a, combiner='sum')