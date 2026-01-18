import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def default_variable_creator_v2(next_creator=None, **kwargs):
    """Default variable creator."""
    assert next_creator is None
    initial_value = kwargs.get('initial_value', None)
    trainable = kwargs.get('trainable', None)
    validate_shape = kwargs.get('validate_shape', True)
    caching_device = kwargs.get('caching_device', None)
    name = kwargs.get('name', None)
    variable_def = kwargs.get('variable_def', None)
    dtype = kwargs.get('dtype', None)
    import_scope = kwargs.get('import_scope', None)
    constraint = kwargs.get('constraint', None)
    distribute_strategy = kwargs.get('distribute_strategy', None)
    synchronization = kwargs.get('synchronization', None)
    aggregation = kwargs.get('aggregation', None)
    shape = kwargs.get('shape', None)
    experimental_enable_variable_lifting = kwargs.get('experimental_enable_variable_lifting', None)
    return ResourceVariable(initial_value=initial_value, trainable=trainable, validate_shape=validate_shape, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, variable_def=variable_def, import_scope=import_scope, distribute_strategy=distribute_strategy, synchronization=synchronization, aggregation=aggregation, shape=shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting)