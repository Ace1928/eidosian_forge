import collections
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use
def register_resource(handle, create_op, is_initialized_op, is_shared=True):
    """Registers a resource into the appropriate collections.

  This makes the resource findable in either the shared or local resources
  collection.

  Args:
   handle: op which returns a handle for the resource.
   create_op: op which initializes the resource.
   is_initialized_op: op which returns a scalar boolean tensor of whether
    the resource has been initialized.
   is_shared: if True, the resource gets added to the shared resource
    collection; otherwise it gets added to the local resource collection.

  """
    resource = _Resource(handle, create_op, is_initialized_op)
    if is_shared:
        ops.add_to_collection(ops.GraphKeys.RESOURCES, resource)
    else:
        ops.add_to_collection(ops.GraphKeys.LOCAL_RESOURCES, resource)