import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export
@save_as_bf16.setter
def save_as_bf16(self, save_as_bf16):
    """Enables saving float32 as bfloat16."""
    self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32