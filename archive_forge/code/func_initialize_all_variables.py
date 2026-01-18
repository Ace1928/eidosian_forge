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
@tf_export(v1=['initialize_all_variables'])
@tf_should_use.should_use_result
@deprecated('2017-03-02', 'Use `tf.global_variables_initializer` instead.')
def initialize_all_variables():
    """See `tf.compat.v1.global_variables_initializer`."""
    return global_variables_initializer()