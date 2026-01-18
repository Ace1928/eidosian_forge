import functools
import tensorflow as tf
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