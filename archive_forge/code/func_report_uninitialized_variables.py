import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['report_uninitialized_variables'])
@tf_should_use.should_use_result
def report_uninitialized_variables(var_list=None, name='report_uninitialized_variables'):
    """Adds ops to list the names of uninitialized variables.

  When run, it returns a 1-D tensor containing the names of uninitialized
  variables if there are any, or an empty array if there are none.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the value of
      `global_variables() + local_variables()`
    name: Optional name of the `Operation`.

  Returns:
    A 1-D tensor containing names of the uninitialized variables, or an empty
    1-D tensor if there are no variables or no uninitialized variables.
  """
    if var_list is None:
        var_list = global_variables() + local_variables()
        if not var_list:
            var_list = []
            for op in ops.get_default_graph().get_operations():
                if op.type in ['Variable', 'VariableV2', 'AutoReloadVariable']:
                    var_list.append(op.outputs[0])
    with ops.name_scope(name):
        if var_list:
            init_vars = [state_ops.is_variable_initialized(v) for v in var_list]
        local_device = os.environ.get('TF_DEVICE_FOR_UNINITIALIZED_VARIABLE_REPORTING', '/cpu:0')
        with ops.device(local_device):
            if not var_list:
                return array_ops.constant([], dtype=dtypes.string)
            else:
                variables_mask = math_ops.logical_not(array_ops_stack.stack(init_vars))
                variable_names_tensor = array_ops.constant([s.op.name for s in var_list])
                return array_ops.boolean_mask(variable_names_tensor, variables_mask)