import functools
import tensorflow as tf
def indexed_slices_intersection_indices_and_values(x1, x2):
    """Compute the indices for the intersection of two `tf.IndexedSlices` and
    modify the values for these indices.

    Args:
        x1: the first `tf.IndexedSlices`.
        x2: the second `tf.IndexedSlices`.
    Returns: A tuple containing:
        - the indices for the intersection
        - `x1` values for the intersection indices (some values were removed)
        - `x2` values for the intersection indices (some values were removed)
    """
    dim_0 = x1.dense_shape[0]
    x1_indices_expanded = tf.expand_dims(x1.indices, axis=1)
    x2_indices_expanded = tf.expand_dims(x2.indices, axis=1)
    x1_indices_count = x1_indices_expanded.shape[0]
    x2_indices_count = x2_indices_expanded.shape[0]
    x1_indices_one_hot = tf.scatter_nd(x1_indices_expanded, ones_bool((x1_indices_count,)), (dim_0,))
    x2_indices_one_hot = tf.scatter_nd(x2_indices_expanded, ones_bool((x2_indices_count,)), (dim_0,))
    intersection_indices = tf.squeeze(tf.where(tf.math.logical_and(x1_indices_one_hot, x2_indices_one_hot)), axis=-1)
    intersection_indices_count = tf.shape(intersection_indices)[0]

    def empty_intersection():
        return (intersection_indices, empty_tensor((0,) + x1.values.shape[1:], x1.dtype), empty_tensor((0,) + x2.values.shape[1:], x2.dtype))

    def non_empty_intersection():

        def values_for_intersection(indices_expanded, indices_count, values):
            indices_indices = tf.scatter_nd(indices_expanded, tf.range(indices_count), (dim_0,))
            to_intersection_indices = tf.gather(indices_indices, intersection_indices)
            return tf.gather(values, to_intersection_indices)
        x1_values_for_intersection = tf.cond(tf.equal(x1_indices_count, intersection_indices_count), lambda: x1.values, lambda: values_for_intersection(x1_indices_expanded, x1_indices_count, x1.values))
        x2_values_for_intersection = tf.cond(tf.equal(x2_indices_count, intersection_indices_count), lambda: x2.values, lambda: values_for_intersection(x2_indices_expanded, x2_indices_count, x2.values))
        return (intersection_indices, x1_values_for_intersection, x2_values_for_intersection)
    return tf.cond(tf.equal(intersection_indices_count, 0), empty_intersection, non_empty_intersection)