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
@dispatch.dispatch_for_api(string_ops.string_join)
def string_join(inputs: typing.List[ragged_tensor.RaggedOrDense], separator='', name=None):
    """RaggedTensor implementation for tf.strings.join."""
    if len(inputs) < 0:
        raise ValueError('tf.strings.join: expected at least one input.')
    with ops.name_scope(name, 'RaggedStringJoin', inputs):
        return ragged_functional_ops.map_flat_values(string_ops.string_join, inputs, separator)