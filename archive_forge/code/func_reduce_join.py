import typing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(string_ops.reduce_join_v2)
def reduce_join(inputs: ragged_tensor.Ragged, axis=None, keepdims=None, separator='', name=None):
    """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
    return ragged_math_ops.ragged_reduce_aggregate(string_ops.reduce_join, string_ops.unsorted_segment_join, inputs, axis, keepdims, separator, name or 'RaggedSegmentJoin')