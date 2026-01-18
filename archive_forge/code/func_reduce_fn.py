import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
def reduce_fn(state, value):
    spec, component = value
    if isinstance(spec, internal.TensorSpec):
        try:
            component = ops.convert_to_tensor(component, spec.dtype)
        except (TypeError, ValueError):
            raise ValueError(f'Value {component} is not convertible to a tensor with dtype {spec.dtype} and shape {spec.shape}.')
        if not component.shape.is_compatible_with(spec.shape):
            raise ValueError(f'Value {component} is not convertible to a tensor with dtype {spec.dtype} and shape {spec.shape}.')
    return encode_fn(state, spec, component)