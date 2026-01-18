import functools
import tensorflow as tf
def non_empty_intersection():

    def values_for_intersection(indices_expanded, indices_count, values):
        indices_indices = tf.scatter_nd(indices_expanded, tf.range(indices_count), (dim_0,))
        to_intersection_indices = tf.gather(indices_indices, intersection_indices)
        return tf.gather(values, to_intersection_indices)
    x1_values_for_intersection = tf.cond(tf.equal(x1_indices_count, intersection_indices_count), lambda: x1.values, lambda: values_for_intersection(x1_indices_expanded, x1_indices_count, x1.values))
    x2_values_for_intersection = tf.cond(tf.equal(x2_indices_count, intersection_indices_count), lambda: x2.values, lambda: values_for_intersection(x2_indices_expanded, x2_indices_count, x2.values))
    return (intersection_indices, x1_values_for_intersection, x2_values_for_intersection)