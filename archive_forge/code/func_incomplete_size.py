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
def incomplete_size(self, name=None):
    """Returns the number of incomplete elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
        name = '%s_incomplete_size' % self._name
    return self._incomplete_size_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)