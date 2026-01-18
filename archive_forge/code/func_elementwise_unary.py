import functools
import tensorflow as tf
def elementwise_unary(func):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    a zero-preserving element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape, and if it is a
      `tf.SparseTensor` or `tf.IndexedSlices`, the indices of the result must be
      the same. Therefore:
        - Reduction operations are not supported (e.g. `mean`).
        - Operations for which the result may be dense (e.g. `reciprocal`), or
          the sparse indices depend on the inputs are not supported (e.g.
          `clip`). This implies that `func(0)` must be 0.

    Additional arguments to the function (besides the input tensor) are
    supported as long as they cannot change the indices of the result. For
    instance,`round` is supported, but `clip` is not supported as
    `clip(x, 1.0, 2.0)` would always return a dense tensor.

    Note that if an input sparse tensor contains zero values, the indices and
    the zero values are preserved.

    Args:
        func: The function to wrap.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

    @functools.wraps(func)
    def sparse_wrapper(x, *args, **kwargs):
        if isinstance(x, tf.SparseTensor):
            return sparse_with_values(x, func(x.values, *args, **kwargs))
        elif isinstance(x, tf.IndexedSlices):
            return tf.IndexedSlices(func(x.values, *args, **kwargs), x.indices, x.dense_shape)
        else:
            return func(x, *args, **kwargs)
    return sparse_wrapper