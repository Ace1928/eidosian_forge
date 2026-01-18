import functools
import tensorflow as tf
def elementwise_binary_intersection(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    an element-wise binary operator such that the indices present in the result
    are the intersection of the indices in the two operand.

    The primary use case for this is the `multiply` operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise.
    - The operator must be binary (two input tensors and one output tensor).
    - Both inputs must be of the same shape or one input must be a scalar.
    - The output must be of the same shape as the (non scalar) inputs.
    - The indices of the output must be the intersection of the indices of the
      inputs. This implies that func(0, x) and func(x, 0) must be 0 for any x.
      As a result, if one operand is dense or a scalar, then the indices are the
      ones from the other operand.

    Additional arguments to the function (besides the input tensors) are not
    supported.

    Note that if the operands contains zero values at some common indices, the
    indices and the zero values are preserved.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x1, x2):
        if isinstance(x1, tf.SparseTensor):
            if isinstance(x2, tf.SparseTensor):
                if x1.indices is x2.indices:
                    return sparse_with_values(x1, func(x1.values, x2.values))
                else:
                    intersection_indices, x1_values_for_intersection, x2_values_for_intersection = sparse_intersection_indices_and_values(x1, x2)
                    output = tf.SparseTensor(intersection_indices, func(x1_values_for_intersection, x2_values_for_intersection), x1.dense_shape)
                    output.set_shape(x1.shape)
                    return output
            elif not hasattr(x2, 'shape') or len(x2.shape) == 0:
                return sparse_with_values(x1, func(x1.values, x2))
            else:
                return sparse_with_values(x1, func(x1.values, tf.gather_nd(x2, x1.indices)))
        elif isinstance(x2, tf.SparseTensor):
            if not hasattr(x1, 'shape') or len(x1.shape) == 0:
                return sparse_with_values(x2, func(x1, x2.values))
            else:
                return sparse_with_values(x2, func(tf.gather_nd(x1, x2.indices), x2.values))
        elif isinstance(x1, tf.IndexedSlices):
            if isinstance(x2, tf.IndexedSlices):
                if x1.indices is x2.indices:
                    return tf.IndexedSlices(func(x1.values, x2.values), x1.indices, x1.dense_shape)
                else:
                    intersection_indices, x1_values_for_intersection, x2_values_for_intersection = indexed_slices_intersection_indices_and_values(x1, x2)
                    return tf.IndexedSlices(func(x1_values_for_intersection, x2_values_for_intersection), intersection_indices, x1.dense_shape)
            elif not hasattr(x2, 'shape') or len(x2.shape) == 0:
                return tf.IndexedSlices(func(x1.values, x2), x1.indices, x1.dense_shape)
            else:
                return tf.IndexedSlices(func(x1.values, tf.gather(x2, x1.indices)), x1.indices, x1.dense_shape)
        elif isinstance(x2, tf.IndexedSlices):
            if not hasattr(x1, 'shape') or len(x1.shape) == 0:
                return tf.IndexedSlices(func(x1, x2.values), x2.indices, x2.dense_shape)
            else:
                return tf.IndexedSlices(func(tf.gather(x1, x2.indices), x2.values), x2.indices, x2.dense_shape)
        return func(x1, x2)
    return sparse_wrapper