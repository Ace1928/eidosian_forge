from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
def repeat_ranges(params, splits, repeats):
    """Repeats each range of `params` (as specified by `splits`) `repeats` times.

  Let the `i`th range of `params` be defined as
  `params[splits[i]:splits[i + 1]]`.  Then this function returns a tensor
  containing range 0 repeated `repeats[0]` times, followed by range 1 repeated
  `repeats[1]`, ..., followed by the last range repeated `repeats[-1]` times.

  Args:
    params: The `Tensor` whose values should be repeated.
    splits: A splits tensor indicating the ranges of `params` that should be
      repeated. Elements should be non-negative integers.
    repeats: The number of times each range should be repeated. Supports
      broadcasting from a scalar value. Elements should be non-negative
      integers.

  Returns:
    A `Tensor` with the same rank and type as `params`.

  #### Example:

  >>> print(repeat_ranges(
  ...     params=tf.constant(['a', 'b', 'c']),
  ...     splits=tf.constant([0, 2, 3]),
  ...     repeats=tf.constant(3)))
  tf.Tensor([b'a' b'b' b'a' b'b' b'a' b'b' b'c' b'c' b'c'],
      shape=(9,), dtype=string)
  """
    splits_checks = [check_ops.assert_non_negative(splits, message="Input argument 'splits' must be non-negative"), check_ops.assert_integer(splits, message=f"Input argument 'splits' must be integer, but got {splits.dtype} instead")]
    repeats_checks = [check_ops.assert_non_negative(repeats, message="Input argument 'repeats' must be non-negative"), check_ops.assert_integer(repeats, message=f"Input argument 'repeats' must be integer, but got {repeats.dtype} instead")]
    splits = control_flow_ops.with_dependencies(splits_checks, splits)
    repeats = control_flow_ops.with_dependencies(repeats_checks, repeats)
    if repeats.shape.ndims != 0:
        repeated_starts = repeat(splits[:-1], repeats, axis=0)
        repeated_limits = repeat(splits[1:], repeats, axis=0)
    else:
        repeated_splits = repeat(splits, repeats, axis=0)
        n_splits = array_ops.shape(repeated_splits, out_type=repeats.dtype)[0]
        repeated_starts = repeated_splits[:n_splits - repeats]
        repeated_limits = repeated_splits[repeats:]
    one = array_ops.ones((), repeated_starts.dtype)
    offsets = gen_ragged_math_ops.ragged_range(repeated_starts, repeated_limits, one)
    return array_ops.gather(params, offsets.rt_dense_values)