import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
def slice_tensorflow_sparse_wrapper(sparse_wrapper, indices):
    from keras.src.utils.module_utils import tensorflow as tf
    if isinstance(indices, slice):
        sparse_indices = sparse_wrapper.ragged_indices[indices]
        sparse_values = sparse_wrapper.ragged_values[indices]
        batch_dim = indices.stop - indices.start
    else:
        sparse_indices = tf.gather(sparse_wrapper.ragged_indices, indices)
        sparse_values = tf.gather(sparse_wrapper.ragged_values, indices)
        if isinstance(indices, list):
            batch_dim = len(indices)
        else:
            batch_dim = indices.shape[0]
            if batch_dim is None:
                batch_dim = tf.shape(indices)[0]
    row_ids = sparse_indices.value_rowids()
    sparse_indices = sparse_indices.flat_values[:, 1:]
    sparse_indices = tf.concat([tf.expand_dims(row_ids, -1), sparse_indices], axis=1)
    sparse_values = sparse_values.flat_values
    sparse_shape = (batch_dim,) + tuple(sparse_wrapper.sparse.shape.as_list()[1:])
    return tf.SparseTensor(sparse_indices, sparse_values, sparse_shape)