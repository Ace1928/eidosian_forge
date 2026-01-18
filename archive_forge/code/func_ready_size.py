import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def ready_size(self, name=None):
    """Compute the number of complete elements in the given barrier.

    Args:
      name: A name for the operation (optional).

    Returns:
      A single-element tensor containing the number of complete elements in the
      given barrier.
    """
    if name is None:
        name = '%s_BarrierReadySize' % self._name
    return gen_data_flow_ops.barrier_ready_size(self._barrier_ref, name=name)