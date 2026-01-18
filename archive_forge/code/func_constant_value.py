import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['ragged.constant_value'])
@dispatch.add_dispatch_support
def constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None, row_splits_dtype='int64'):
    """Constructs a RaggedTensorValue from a nested Python list.

  Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.
  If you wish to construct a constant `RaggedTensor`, use
  [`ragged.constant(...)`](constant.md) instead.

  Example:

  >>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])
  tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),
                       row_splits=array([0, 2, 3, 6]))

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no
  scalar values, then `K` is one greater than the maximum depth of empty lists
  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list` or `tuple` must be a scalar value compatible with `dtype`.
    dtype: `numpy.dtype`.  The type of elements for the returned `RaggedTensor`.
      If not specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensorValue`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensorValue`.  Defaults to `()` if
      `ragged_rank` is not specified.  If `ragged_rank` is specified, then a
      default is chosen based on the contents of `pylist`.
    row_splits_dtype: data type for the constructed `RaggedTensorValue`'s
      row_splits.  One of `numpy.int32` or `numpy.int64`.

  Returns:
    A `tf.RaggedTensorValue` or `numpy.array` with rank `K` and the specified
    `ragged_rank`, containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
    if dtype is not None and isinstance(dtype, dtypes.DType):
        dtype = dtype.as_numpy_dtype
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype).as_numpy_dtype

    def _ragged_factory(values, row_splits):
        row_splits = np.array(row_splits, dtype=row_splits_dtype)
        return ragged_tensor_value.RaggedTensorValue(values, row_splits)

    def _inner_factory(pylist, dtype, shape, name=None):
        return np.reshape(np.array(pylist, dtype=dtype), shape)
    return _constant_value(_ragged_factory, _inner_factory, pylist, dtype, ragged_rank, inner_shape)