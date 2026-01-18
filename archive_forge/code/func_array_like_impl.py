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
def array_like_impl(array_fn, array_like_fn, tensor, dtype, name, optimize=True, layout=None):
    """Internal implementation for ones_like and zeros_like API calls."""
    if not tensor_util.is_tf_type(tensor):
        tensor = ops.convert_to_tensor(tensor, name='tensor')
    tensor_shape = tensor.shape
    tensor_dtype = tensor.dtype
    if context.executing_eagerly():
        if dtype is not None and dtype != tensor_dtype:
            return array_fn(shape_internal(tensor, optimize=optimize), dtype=dtype, name=name, layout=layout)
        return d_api.call_with_layout(array_like_fn, layout, tensor, name=name)
    if optimize and tensor_shape.is_fully_defined() and (tensor_dtype != dtypes.variant):
        return array_fn(tensor_shape, dtype=dtype or tensor_dtype, name=name, layout=layout)
    if dtype is not None and dtype != tensor_dtype and (dtype != dtypes.variant):
        return array_fn(shape_internal(tensor, optimize=optimize), dtype=dtype, name=name, layout=layout)
    return d_api.call_with_layout(array_like_fn, layout, tensor, name=name)