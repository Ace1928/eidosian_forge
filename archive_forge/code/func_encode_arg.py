import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def encode_arg(arg, path):
    """A representation for this argument, for converting into signatures."""
    if isinstance(arg, tensor_lib.Tensor):
        user_specified_name = None
        try:
            user_specified_name = compat.as_str(arg.op.get_attr('_user_specified_name'))
        except (ValueError, AttributeError):
            pass
        if path and user_specified_name and (user_specified_name != path[0]):
            name = user_specified_name
        else:
            name = tensor_lib.sanitize_spec_name('_'.join((str(p) for p in path)))
        return tensor_lib.TensorSpec(arg.shape, arg.dtype, name)
    if isinstance(arg, resource_variable_ops.ResourceVariable):
        return trace_type.from_value(arg, signature_context)
    if isinstance(arg, composite_tensor.CompositeTensor):
        return arg._type_spec
    if isinstance(arg, (int, float, bool, str, type(None), dtypes.DType, tensor_lib.TensorSpec, type_spec.TypeSpec)):
        return arg
    return UnknownArgument()