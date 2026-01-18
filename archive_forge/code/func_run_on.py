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
@tf_export('experimental.dtensor.run_on', v1=[])
@deprecation.deprecated(None, 'Use `dtensor.default_mesh` scope instead.')
@contextlib.contextmanager
def run_on(mesh: layout_lib.Mesh):
    """Runs enclosed functions in the DTensor device scope.

  This function returns a scope. All the ops and tf.functions in this scope will
  run on the DTensor device using the mesh provided.
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants be replicated as DTensor.

  Args:
    mesh: A Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the DTensor device.
  """
    with default_mesh(mesh):
        yield