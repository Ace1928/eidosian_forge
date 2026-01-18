import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
def internal_convert_to_tensor_or_composite(value, dtype=None, name=None, as_ref=False) -> List[Union[tensor_lib.Tensor, internal.IndexedSlices]]:
    """Converts the given object to a `Tensor` or `CompositeTensor`.

  If `value` is a `CompositeTensor` it is returned unmodified.  Otherwise, it
  is converted to a `Tensor` using `convert_to_tensor()`.

  Args:
    value: A `CompositeTensor`, or an object that can be consumed by
      `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `CompositeTensor`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A `Tensor` or `CompositeTensor`, based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
    if isinstance(value, composite_tensor.CompositeTensor):
        value_dtype = getattr(value, 'dtype', None)
        if dtype and (not dtypes.as_dtype(dtype).is_compatible_with(value_dtype)):
            raise ValueError(f'Tensor conversion dtype mismatch. Requested dtype is {dtypes.as_dtype(dtype).name}, Tensor has dtype {value.dtype.name}: {value!r}')
        return value
    else:
        return convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref, accepted_result_types=(tensor_lib.Tensor, composite_tensor.CompositeTensor))