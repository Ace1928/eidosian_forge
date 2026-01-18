import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('summary.experimental.summary_scope', v1=[])
@tf_contextlib.contextmanager
def summary_scope(name, default_name='summary', values=None):
    """Experimental context manager for use when defining a custom summary op.

  This behaves similarly to `tf.name_scope`, except that it returns a generated
  summary tag in addition to the scope name. The tag is structurally similar to
  the scope name - derived from the user-provided name, prefixed with enclosing
  name scopes if any - but we relax the constraint that it be uniquified, as
  well as the character set limitation (so the user-provided name can contain
  characters not legal for scope names; in the scope name these are removed).

  This makes the summary tag more predictable and consistent for the user.

  For example, to define a new summary op called `my_op`:

  ```python
  def my_op(name, my_value, step):
    with tf.summary.summary_scope(name, "MyOp", [my_value]) as (tag, scope):
      my_value = tf.convert_to_tensor(my_value)
      return tf.summary.write(tag, my_value, step=step)
  ```

  Args:
    name: string name for the summary.
    default_name: Optional; if provided, used as default name of the summary.
    values: Optional; passed as `values` parameter to name_scope.

  Yields:
    A tuple `(tag, scope)` as described above.
  """
    name = name or default_name
    current_scope = ops.get_name_scope()
    tag = current_scope + '/' + name if current_scope else name
    name = _INVALID_SCOPE_CHARACTERS.sub('', name) or None
    with ops.name_scope(name, default_name, values, skip_on_eager=False) as scope:
        yield (tag, scope)