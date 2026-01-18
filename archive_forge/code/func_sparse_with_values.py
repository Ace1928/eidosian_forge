import functools
import tensorflow as tf
def sparse_with_values(x, values):
    x_shape = x.shape
    x = tf.SparseTensor(x.indices, values, x.dense_shape)
    x.set_shape(x_shape)
    return x