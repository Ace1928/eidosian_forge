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
@tf_export(v1=['quantize_v2'])
@dispatch.add_dispatch_support
@deprecation.deprecated('2017-10-25', '`tf.quantize_v2` is deprecated, please use `tf.quantization.quantize` instead.')
def quantize_v2(input, min_range, max_range, T, mode='MIN_COMBINED', name=None, round_mode='HALF_AWAY_FROM_ZERO', narrow_range=False, axis=None, ensure_minimum_range=0.01):
    if axis is None:
        axis = -1
    elif axis < 0:
        if input.shape.ndims is None:
            raise ValueError('input should have known rank to use negative axis.')
        axis %= input.shape.ndims
    if ensure_minimum_range != 0.01:
        return gen_array_ops.quantize_v2(input, min_range, max_range, T=T, mode=mode, name=name, round_mode=round_mode, narrow_range=narrow_range, axis=axis, ensure_minimum_range=ensure_minimum_range)
    return gen_array_ops.quantize_v2(input, min_range, max_range, T=T, mode=mode, name=name, round_mode=round_mode, narrow_range=narrow_range, axis=axis)