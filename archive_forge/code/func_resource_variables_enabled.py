import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['resource_variables_enabled'])
def resource_variables_enabled():
    """Returns `True` if resource variables are enabled.

  Resource variables are improved versions of TensorFlow variables with a
  well-defined memory model. Accessing a resource variable reads its value, and
  all ops which access a specific read value of the variable are guaranteed to
  see the same value for that tensor. Writes which happen after a read (by
  having a control or data dependency on the read) are guaranteed not to affect
  the value of the read tensor, and similarly writes which happen before a read
  are guaranteed to affect the value. No guarantees are made about unordered
  read/write pairs.

  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0
  feature.
  """
    global _DEFAULT_USE_RESOURCE
    return _DEFAULT_USE_RESOURCE