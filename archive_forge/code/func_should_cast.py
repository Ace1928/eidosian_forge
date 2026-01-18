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
def should_cast(self, v):
    """Returns True if v has float32 dtype and is intructed to save as bf16.

    Args:
      v : The variable that determines whether to cast.

    Returns:
      True if current savable DVariable is instructed to save as bfloat16 and
        the variable has dtype float32.
    """
    return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32