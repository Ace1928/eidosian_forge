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
@tf_export('math.count_nonzero', v1=[])
@dispatch.add_dispatch_support
def count_nonzero_v2(input, axis=None, keepdims=None, dtype=dtypes.int64, name=None):
    """Computes number of nonzero elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  **NOTE** Floating point comparison to zero is done by exact floating point
  equality check.  Small values are **not** rounded to zero for purposes of
  the nonzero check.

  For example:

  ```python
  x = tf.constant([[0, 1, 0], [1, 1, 0]])
  tf.math.count_nonzero(x)  # 3
  tf.math.count_nonzero(x, 0)  # [1, 2, 0]
  tf.math.count_nonzero(x, 1)  # [1, 2]
  tf.math.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.math.count_nonzero(x, [0, 1])  # 3
  ```

  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.

  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.math.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```

  Args:
    input: The tensor to reduce. Should be of numeric type, `bool`, or `string`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input), rank(input))`.
    keepdims: If true, retains reduced dimensions with length 1.
    dtype: The output dtype; defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor (number of nonzero values).
  """
    if keepdims is None:
        keepdims = False
    with ops.name_scope(name, 'count_nonzero', [input]):
        input = ops.convert_to_tensor(input, name='input')
        if input.dtype == dtypes.bool:
            predicate = input
        else:
            zero = array_ops.zeros([], dtype=input.dtype)
            predicate = gen_math_ops.not_equal(input, zero)
        return cast(reduce_sum(cast(predicate, dtypes.int64), axis=axis, keepdims=keepdims), dtype=dtype)