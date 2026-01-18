import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input, axis, keepdims, separator=None, name=None):
    """Aggregates across axes of a RaggedTensor using the given `Tensor` ops.

  Reduces `rt_input` along the dimensions given in `axis`.  The rank of the
  tensor is reduced by 1 for each entry in `axis`.  If `axis` is not specified,
  then all dimensions are reduced, and a scalar value is returned.

  This op assumes that `reduce_op` and `unsorted_segment_op` are associative;
  if not, then reducing multiple axes will return incorrect results.  (In
  particular, reducing multiple axes is currently implemented by reducing the
  axes one at a time.)

  Args:
    reduce_op: The tensorflow `op` that should be used to reduce values in
      uniform dimensions.  Must have the same signature and basic behavior as
      `reduce_sum`, `reduce_max`, etc.
    unsorted_segment_op: The tensorflow `op` that should be used to combine
      values in ragged dimensions.  Must have the same signature and basic
      behavior as `unsorted_segment_sum`, `unsorted_segment_max`, etc.
    rt_input: A `Tensor` or `RaggedTensor` containing the values to be reduced.
    axis: The axis or axes to reduce.  May be `None` (to reduce all axes), an
      `int` (to reduce a single axis), a `list` or `tuple` of `int` (to reduce a
      given set of axes), or a `Tensor` with a constant value.  Must be in the
      range `[0, rt_input.rank)`.
    keepdims: If true, retains reduced dimensions with length 1.
    separator: An optional string. Defaults to None. The separator to use when
      joining. The separator must not be set for non-string data types. (i.e. if
      separator is not None then it uses string ops)
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the reduced values.  The returned tensor
    has the same dtype as `data`, and its shape is given by removing the
    dimensions specified in `axis` from `rt_input.shape`.  The `ragged_rank`
    of the returned tensor is given by substracting any ragged dimensions
    specified in `axis` from `rt_input.ragged_rank`.
  Raises:
    ValueError: If `axis` contains a `Tensor` whose value is not constant.
  """
    if separator is None:
        maybe_separator = {}
    else:
        maybe_separator = {'separator': separator}
    if not ragged_tensor.is_ragged(rt_input):
        return reduce_op(rt_input, axis, keepdims=keepdims, name=name, **maybe_separator)
    if isinstance(axis, tensor.Tensor):
        axis = tensor_util.constant_value(axis)
        if axis is None:
            raise ValueError('axis must be known at graph construction time.')
        if isinstance(axis, np.ndarray):
            axis = axis.tolist()
    if axis is None:
        result = reduce_op(rt_input.flat_values, None, keepdims=keepdims, name=name, **maybe_separator)
        if keepdims:
            for _ in rt_input.shape[1:]:
                result = array_ops.expand_dims(result, axis=0)
        return result
    with ops.name_scope(name, 'RaggedReduce', [rt_input, axis]):
        if isinstance(axis, (tuple, list)):
            if not axis:
                return rt_input
            elif len(axis) == 1:
                axis = axis[0]
            else:
                axis = [array_ops.get_positive_axis(a, rt_input.shape.ndims, 'axis[%s]' % i, 'rank(input_tensor)') for i, a in enumerate(axis)]
                axis = sorted(axis)
                inner_reduced = ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input, axis[-1], keepdims, separator)
                return ragged_reduce_aggregate(reduce_op, unsorted_segment_op, inner_reduced, axis[:-1], keepdims, separator)
        rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input, name='rt_input')
        axis = array_ops.get_positive_axis(axis, rt_input.shape.ndims, ndims_name='rank(input_tensor)')
        if axis == 0:
            row_lengths = rt_input.row_splits[1:] - rt_input.row_splits[:-1]
            num_segments = math_ops.maximum(math_ops.reduce_max(row_lengths), 0)
            segment_ids = range(row_lengths).values
            result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values, segment_ids, num_segments, separator)
            if keepdims:
                result = array_ops.expand_dims(result, axis=0)
            return result
        elif axis == 1:
            num_segments = array_ops.shape(rt_input.row_splits)[0] - 1
            segment_ids = segment_id_ops.row_splits_to_segment_ids(rt_input.row_splits)
            result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values, segment_ids, num_segments, separator)
            if keepdims:
                result = array_ops.expand_dims(result, axis=1)
            return result
        else:
            return rt_input.with_values(ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input.values, axis - 1, keepdims, separator))