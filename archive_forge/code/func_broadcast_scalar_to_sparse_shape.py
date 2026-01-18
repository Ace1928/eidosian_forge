import functools
import tensorflow as tf
def broadcast_scalar_to_sparse_shape(scalar, sparse):
    output = tf.broadcast_to(scalar, sparse.dense_shape)
    output.set_shape(sparse.shape)
    return output