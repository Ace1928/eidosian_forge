import functools
import tensorflow as tf
def elementwise_division(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    element-wise binary division and related operators.

    This decorator is designed for operations related to the division of two
    operands (e.g. `divide`). It accepts `tf.SparseTensor` and
    `tf.IndexedSlices` for both the dividend and the divisor, but handles them
    differently based on whether they are the dividend or the divisor.

    - If the divisor is a `tf.SparseTensor` or `tf.IndexedSlices`, it is
      densified and the result is dense because the result contains Inf or Nan
      outside of the indices of the dividend.
    - If the dividend is a `tf.SparseTensor` or `tf.IndexedSlices` and the
      divisor is dense, it finds occurrences of zeros and NaNs in the divisor.
      The result may therefore have more indices than there were in the dividend
      to return correct values where the divisor was zero or NaN.
    - If the dividend is a `tf.SparseTensor` or `tf.IndexedSlices` and the
      divisor is a scalar, it does the division element-wise. Note that the
      result is incorrectly sparse if the scalar divisor is zero.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x1, x2):
        if isinstance(x1, tf.SparseTensor):
            if isinstance(x2, tf.SparseTensor):
                x1 = sparse_to_dense(x1)
                x2 = sparse_to_dense(x2)
            elif not hasattr(x2, 'shape') or len(x2.shape) == 0:
                return sparse_with_values(x1, func(x1.values, x2))
            else:
                x2_zeros_and_nans = tf.equal(x2, 0)
                if not tf.as_dtype(x2.dtype).is_integer:
                    x2_zeros_and_nans = tf.math.logical_or(x2_zeros_and_nans, tf.math.is_nan(x2))

                def func_for_x1_indices():
                    return sparse_with_values(x1, func(x1.values, tf.gather_nd(x2, x1.indices)))

                def func_for_union_indices():
                    x2_zeros_and_nan_indices = tf.where(x2_zeros_and_nans)
                    union_indices, x1_values_for_union, _ = sparse_union_indices_and_values(x1, x2_zeros_and_nan_indices)
                    output = tf.SparseTensor(union_indices, func(x1_values_for_union, tf.gather_nd(x2, union_indices)), x1.dense_shape)
                    output.set_shape(x1.shape)
                    return output
                return tf.cond(tf.reduce_any(x2_zeros_and_nans), func_for_union_indices, func_for_x1_indices)
        elif isinstance(x2, tf.SparseTensor):
            x2 = sparse_to_dense(x2)
        elif isinstance(x1, tf.IndexedSlices):
            if isinstance(x2, tf.IndexedSlices):
                x1 = tf.convert_to_tensor(x1)
                x2 = tf.convert_to_tensor(x2)
            elif not hasattr(x2, 'shape') or len(x2.shape) == 0:
                return tf.IndexedSlices(func(x1.values, x2), x1.indices, x1.dense_shape)
            else:
                x2_zeros_and_nans = tf.equal(x2, 0)
                if not tf.as_dtype(x2.dtype).is_integer:
                    x2_zeros_and_nans = tf.math.logical_or(x2_zeros_and_nans, tf.math.is_nan(x2))
                x2_zeros_and_nans = tf.reduce_any(x2_zeros_and_nans, axis=tuple(range(1, x2.shape.rank)))

                def func_for_x1_indices():
                    return tf.IndexedSlices(func(x1.values, tf.gather(x2, x1.indices)), x1.indices, x1.dense_shape)

                def func_for_union_indices():
                    x2_zeros_and_nan_indices = tf.squeeze(tf.where(x2_zeros_and_nans), axis=-1)
                    union_indices, x1_values_for_union, _ = indexed_slices_union_indices_and_values(x1, x2_zeros_and_nan_indices)
                    return tf.IndexedSlices(func(x1_values_for_union, tf.gather(x2, union_indices)), union_indices, x1.dense_shape)
                return tf.cond(tf.reduce_any(x2_zeros_and_nans), func_for_union_indices, func_for_x1_indices)
        elif isinstance(x2, tf.IndexedSlices):
            x2 = tf.convert_to_tensor(x2)
        return func(x1, x2)
    return sparse_wrapper