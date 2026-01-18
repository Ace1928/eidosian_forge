import contextlib
import threading
from typing import Any, Callable, Optional, Sequence
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.fetch_layout', v1=[])
def fetch_layout(tensor: tensor_lib.Tensor) -> layout_lib.Layout:
    """Fetches the layout of a DTensor.

  Args:
    tensor: The DTensor whose layout is to be fetched.

  Returns:
    The `Layout` of this DTensor.

  Raises:
    RuntimeError: When not called eagerly.
  """
    return _dtensor_device().fetch_layout(tensor)