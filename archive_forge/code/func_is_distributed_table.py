from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def is_distributed_table(v):
    """Determine if an object is a DistributedTable."""
    return getattr(v, 'is_distributed_table', False)