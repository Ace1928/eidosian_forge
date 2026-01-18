from typing import Generator, Optional, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_bfloat16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == dtypes.bfloat16:
        kwargs['dtype'] = dtypes.float32
        cast_to_bfloat16 = True
    var = getter(*args, **kwargs)
    if cast_to_bfloat16:
        var = math_ops.cast(var, dtypes.bfloat16)
    return var