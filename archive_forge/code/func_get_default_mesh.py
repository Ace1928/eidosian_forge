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
@tf_export('experimental.dtensor.get_default_mesh', v1=[])
def get_default_mesh() -> Optional[layout_lib.Mesh]:
    """Return the default mesh under the current dtensor device context.

  In the case that dtensor device system is not initialized, this function
  will return None.

  Returns:
    The current default mesh for the dtensor device context.
  """
    if _dtensor_singleton is None:
        return None
    else:
        return _dtensor_singleton._current_default_mesh