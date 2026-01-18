import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def concatenate_context_input(context_input, sequence_input):
    """Replicates `context_input` across all timesteps of `sequence_input`.

  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
  This value is appended to `sequence_input` on dimension 2 and the result is
  returned.

  Args:
    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.
    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
      padded_length, d0]`.

  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
    d0 + d1]`.

  Raises:
    ValueError: If `sequence_input` does not have rank 3 or `context_input` does
      not have rank 2.
  """
    seq_rank_check = check_ops.assert_rank(sequence_input, 3, message='sequence_input must have rank 3', data=[array_ops.shape(sequence_input)])
    seq_type_check = check_ops.assert_type(sequence_input, dtypes.float32, message='sequence_input must have dtype float32; got {}.'.format(sequence_input.dtype))
    ctx_rank_check = check_ops.assert_rank(context_input, 2, message='context_input must have rank 2', data=[array_ops.shape(context_input)])
    ctx_type_check = check_ops.assert_type(context_input, dtypes.float32, message='context_input must have dtype float32; got {}.'.format(context_input.dtype))
    with ops.control_dependencies([seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
        padded_length = array_ops.shape(sequence_input)[1]
        tiled_context_input = array_ops.tile(array_ops.expand_dims(context_input, 1), array_ops.concat([[1], [padded_length], [1]], 0))
    return array_ops.concat([sequence_input, tiled_context_input], 2)