import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['reverse_sequence'])
@deprecation.deprecated_args(None, 'seq_dim is deprecated, use seq_axis instead', 'seq_dim')
@deprecation.deprecated_args(None, 'batch_dim is deprecated, use batch_axis instead', 'batch_dim')
def reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None, batch_dim=None):
    """Reverses variable length slices.

  This op first slices `input` along the dimension `batch_axis`, and for
  each slice `i`, reverses the first `seq_lengths[i]` elements along the
  dimension `seq_axis`.

  The elements of `seq_lengths` must obey `seq_lengths[i] <=
  input.dims[seq_axis]`, and `seq_lengths` must be a vector of length
  `input.dims[batch_axis]`.

  The output slice `i` along dimension `batch_axis` is then given by
  input slice `i`, with the first `seq_lengths[i]` slices along
  dimension `seq_axis` reversed.

  Example usage:

  >>> seq_lengths = [7, 2, 3, 5]
  >>> input = [[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
  ...          [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]]
  >>> output = tf.reverse_sequence(input, seq_lengths, seq_axis=1, batch_axis=0)
  >>> output
  <tf.Tensor: shape=(4, 8), dtype=int32, numpy=
  array([[0, 0, 5, 4, 3, 2, 1, 0],
         [2, 1, 0, 0, 0, 0, 0, 0],
         [3, 2, 1, 4, 0, 0, 0, 0],
         [5, 4, 3, 2, 1, 6, 7, 8]], dtype=int32)>

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor`. Must be one of the following types: `int32`,
      `int64`. 1-D with length `input.dims(batch_axis)` and `max(seq_lengths) <=
      input.dims(seq_axis)`
    seq_axis: An `int`. The dimension which is partially reversed.
    batch_axis: An optional `int`. Defaults to `0`. The dimension along which
      reversal is performed.
    name: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as input.
  """
    seq_axis = deprecation.deprecated_argument_lookup('seq_axis', seq_axis, 'seq_dim', seq_dim)
    batch_axis = deprecation.deprecated_argument_lookup('batch_axis', batch_axis, 'batch_dim', batch_dim)
    return gen_array_ops.reverse_sequence(input=input, seq_lengths=seq_lengths, seq_dim=seq_axis, batch_dim=batch_axis, name=name)