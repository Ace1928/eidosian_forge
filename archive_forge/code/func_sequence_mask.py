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
@tf_export('sequence_mask')
@dispatch.add_dispatch_support
def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
    """Returns a mask tensor representing the first N positions of each cell.

  If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
  dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with

  ```
  mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
  ```

  Examples:

  ```python
  tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                  #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]

  tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                    #   [True, True, True]],
                                    #  [[True, True, False],
                                    #   [False, False, False]]]
  ```

  Args:
    lengths: integer tensor, all its values <= maxlen.
    maxlen: scalar integer tensor, size of last dimension of returned tensor.
      Default is the maximum value in `lengths`.
    dtype: output type of the resulting tensor.
    name: name of the op.

  Returns:
    A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
  Raises:
    ValueError: if `maxlen` is not a scalar.
  """
    with ops.name_scope(name, 'SequenceMask', [lengths, maxlen]):
        lengths = ops.convert_to_tensor(lengths)
        if maxlen is None:
            maxlen = gen_math_ops._max(lengths, _all_dimensions(lengths))
            maxlen = gen_math_ops.maximum(constant(0, maxlen.dtype), maxlen)
        else:
            maxlen = ops.convert_to_tensor(maxlen)
        if maxlen.get_shape().ndims is not None and maxlen.get_shape().ndims != 0:
            raise ValueError(f"Argument `maxlen` must be scalar for sequence_mask, received `maxlen` = {maxlen} with shape '{maxlen.get_shape()}' instead")
        row_vector = gen_math_ops._range(constant(0, maxlen.dtype), maxlen, constant(1, maxlen.dtype))
        matrix = gen_math_ops.cast(expand_dims(lengths, -1), maxlen.dtype)
        result = row_vector < matrix
        if dtype is None or result.dtype.is_compatible_with(dtype):
            return result
        else:
            return gen_math_ops.cast(result, dtype)