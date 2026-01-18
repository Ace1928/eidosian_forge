import functools
import tensorflow as tf
def sparse_intersection_indices_and_values(x1, x2):
    """Compute the indices for the intersection of two `tf.SparseTensor`s and
    modify the values for these indices.

    Args:
        x1: the first `tf.SparseTensor`.
        x2: the second `tf.SparseTensor`.
    Returns: A tuple containing:
        - the indices for the intersection
        - `x1` values for the intersection indices (some values were removed)
        - `x2` values for the intersection indices (some values were removed)
    """
    ones1 = tf.sparse.map_values(ones_like_int8, x1)
    ones2 = tf.sparse.map_values(ones_like_int8, x2)
    intersection_extra_dim = tf.sets.intersection(tf.sparse.expand_dims(ones1, axis=-1), tf.sparse.expand_dims(ones2, axis=-1))

    def empty_intersection():
        return (empty_tensor((0, x1.shape.rank), dtype=tf.int64), empty_tensor((0,), dtype=x1.values.dtype), empty_tensor((0,), dtype=x2.values.dtype))

    def non_empty_intersection():
        intersection = tf.sparse.reshape(intersection_extra_dim, x1.dense_shape)
        zeros1 = tf.sparse.map_values(zeros_like_int8, x1)
        zeros2 = tf.sparse.map_values(zeros_like_int8, x2)
        mask1 = tf.sparse.add(zeros1, intersection)
        mask2 = tf.sparse.add(zeros2, intersection)
        return (intersection.indices, tf.sparse.retain(x1, tf.cast(mask1.values, tf.bool)).values, tf.sparse.retain(x2, tf.cast(mask2.values, tf.bool)).values)
    return tf.cond(tf.equal(tf.size(intersection_extra_dim), 0), empty_intersection, non_empty_intersection)