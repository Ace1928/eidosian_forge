import functools
import tensorflow as tf
def func_for_union_indices():
    x2_zeros_and_nan_indices = tf.squeeze(tf.where(x2_zeros_and_nans), axis=-1)
    union_indices, x1_values_for_union, _ = indexed_slices_union_indices_and_values(x1, x2_zeros_and_nan_indices)
    return tf.IndexedSlices(func(x1_values_for_union, tf.gather(x2, union_indices)), union_indices, x1.dense_shape)