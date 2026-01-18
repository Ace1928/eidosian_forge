import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def maybe_promote_tensors(*tensors, force_same_dtype=False):
    """Promotes tensors if numpy style promotion is enabled.

  This function promotes `tensors` according to numpy promotion rules
  if numpy style promotion is enabled.  Otherwise, if
  `force_same_dtype` is `True`, it force-casts `tensors[1:]` to
  `tensor[0]`'s dtype. Note that this force-cast can be problematic.
  For example, when some `tensors[1:]` elements can be silently
  downcasted.

  Args:
    *tensors: the list of tensors to promote.
    force_same_dtype: bool (optional, default to `False`). When numpy
      style promotion is disabled and `force_same_dtype` is `True`,
      this function will force-casts `tensors[1:]` to `tensor[0]`'s
      dtype (which could be problematic).

  Returns:
    The promoted list of tensors.
  """
    if ops.is_auto_dtype_conversion_enabled():
        return tensors
    if not tensors:
        return tensors
    if not ops.is_numpy_style_type_promotion():
        if not force_same_dtype:
            return tensors
        promoted_tensors = []
        promoted_tensors.append(tensors[0])
        dtype = tensors[0].dtype.base_dtype
        for tensor in tensors[1:]:
            promoted_tensors.append(ops.convert_to_tensor(tensor, dtype, name='x'))
        return promoted_tensors
    result_type = np_dtypes._result_type(*[_maybe_get_dtype(x) for x in nest.flatten(tensors)])

    def _promote_or_cast(x):
        if isinstance(x, tensor_lib.Tensor):
            x = cast(x, result_type)
        else:
            x = ops.convert_to_tensor(x, result_type)
        return x
    return [_promote_or_cast(x) for x in tensors]