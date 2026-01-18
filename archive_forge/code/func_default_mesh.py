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
@tf_export('experimental.dtensor.default_mesh', v1=[])
@contextlib.contextmanager
def default_mesh(mesh: layout_lib.Mesh):
    """Sets the default DTensor device mesh to use for enclosed functions.

  This function returns a scope. All the ops and tf.functions in this scope will
  default to this DTensor mesh if a mesh cannot be inferred from any of the
  inputs
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants are replicated as DTensors.

  Args:
    mesh: A Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the given mesh.
  """
    if not isinstance(mesh, layout_lib.Mesh):
        raise ValueError(f'Expect `mesh` to be `Mesh`, got {type(mesh)}')
    with _dtensor_device()._experimental_default_mesh(mesh):
        with ops.device(device_name()):
            yield