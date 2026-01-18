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
@dispatch.dispatch_for_api(math_ops.cumsum)
def ragged_cumsum(x: ragged_tensor.Ragged, axis: int=0, exclusive: bool=False, reverse: bool=False, name: typing.Optional[str]=None):
    """Calculate math_ops.cumsum for a RaggedTensor.

  Given a ragged tensor `x`, the `result` is a ragged tensor with the same
  shape. One can calculate the value of `result[i_1...i_k]` as follows:
  ```
  dense_result=tf.math.cumsum(rt.to_tensor(), axis=axis, exclusive=exclusive,
                              reverse=reverse)
  result[i_1...i_k]=dense_result[i_1...i_k]
  ```

  Args:
    x: the original ragged tensor to sum.
    axis: the axis along which to sum, can range -rank<=axis<rank.
    exclusive: is the sum exclusive or inclusive? If True, then result[0]=0.
        If False, then result[0]=x[0].
    reverse: If True, sum from back to front.
    name: the name of the op.
  Returns:
    the cumulative sum.
  """
    with ops.name_scope(name, 'RaggedCumSum', [x, axis, exclusive, reverse]):
        axis = array_ops.get_positive_axis(axis, x.shape.rank, ndims_name='rank')
        if axis == x.ragged_rank:
            last_rp = x._nested_row_partitions[-1]
            return x.with_flat_values(_cumsum_flat_values_at_ragged_rank(last_rp, x.flat_values, exclusive=exclusive, reverse=reverse))
        elif axis > x.ragged_rank:
            new_axis = axis - x.ragged_rank
            cumsum_bound = functools.partial(math_ops.cumsum, axis=new_axis, exclusive=exclusive, reverse=reverse)
            return ragged_functional_ops.map_flat_values(cumsum_bound, x)
        else:
            dense_version = x.to_tensor()
            result = math_ops.cumsum(dense_version, axis, exclusive=exclusive, reverse=reverse, name=name)
            return ragged_tensor.RaggedTensor.from_tensor(result, lengths=x.nested_row_lengths())