import numpy as np
from tensorflow.core.framework import resource_handle_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def get_raw_handle(self):
    """Return the raw handle of the tensor.

    Note that the method disables the automatic garbage collection of this
    persistent tensor. The caller is now responsible for managing the life
    time of the tensor.
    """
    self._auto_gc_enabled = False
    return self._handle