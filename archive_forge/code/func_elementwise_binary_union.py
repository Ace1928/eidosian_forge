import functools
import tensorflow as tf
def elementwise_binary_union(sparse_op, densify_mixed=False):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    an element-wise binary operator such that the indices present in the result
    are the union of the indices in the two operand.

    The primary use case for this is the `add` and `subtract` operators.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise.
    - The operator must be binary (two input tensors and one output tensor).
    - Both inputs must be of the same shape or one input must be a scalar.
    - The output must be of the same shape as the (non scalar) inputs.
    - The indices of the output must be the union of the indices of the inputs.
      This implies that func(0, 0) must be 0. As a result, if one operand is
      dense or a scalar, then the result will be dense.

    Additional arguments to the function (besides the input tensors) are not
    supported.

    Note that if the result of the operation is zero at some indices, including
    because the operands were zero at these indices, the zeros and indices are
    preserved.

    Args:
        sparse_op: implementation of the operation for `tf.SparseTensor`. Must
            work if both of the operands are `tf.SparseTensor`s and can
            optionally work if one of the operand is a `tf.SparseTensor` and
            the other one is dense tensor, see `densify_mixed`.
        densify_mixed: if `True`, `sparse_op` does not support a mix of
            `tf.SparseTensor` and dense tensor or dense tensor with
            `tf.SparseTensor` and the `tf.SparseTensor` tensor is densified.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    def wrap_elementwise_binary_union(func):

        @functools.wraps(func)
        def sparse_wrapper(x1, x2):
            if isinstance(x1, tf.SparseTensor):
                if isinstance(x2, tf.SparseTensor):
                    if x1.indices is x2.indices:
                        return sparse_with_values(x1, func(x1.values, x2.values))
                    else:
                        output = sparse_op(x1, x2)
                        output.set_shape(x1.shape)
                        return output
                elif densify_mixed:
                    x1 = sparse_to_dense(x1)
                else:
                    if not hasattr(x2, 'shape') or len(x2.shape) == 0:
                        x2 = broadcast_scalar_to_sparse_shape(x2, x1)
                    return sparse_op(x1, x2)
            elif isinstance(x2, tf.SparseTensor):
                if densify_mixed:
                    x2 = sparse_to_dense(x2)
                else:
                    if not hasattr(x1, 'shape') or len(x1.shape) == 0:
                        x1 = broadcast_scalar_to_sparse_shape(x1, x2)
                    return sparse_op(x1, x2)
            elif isinstance(x1, tf.IndexedSlices):
                if isinstance(x2, tf.IndexedSlices):
                    if x1.indices is x2.indices:
                        return tf.IndexedSlices(func(x1.values, x2.values), x1.indices, x1.dense_shape)
                    else:
                        union_indices, x1_values_for_union, x2_values_for_union = indexed_slices_union_indices_and_values(x1, x2.indices, x2.values)
                        return tf.IndexedSlices(func(x1_values_for_union, x2_values_for_union), union_indices, x1.dense_shape)
                else:
                    x1 = tf.convert_to_tensor(x1)
            elif isinstance(x2, tf.IndexedSlices):
                x2 = tf.convert_to_tensor(x2)
            return func(x1, x2)
        return sparse_wrapper
    return wrap_elementwise_binary_union