from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.set_device_policy')
def set_device_policy(device_policy):
    """Sets the current thread device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  When using the default, an appropriate policy will be picked automatically.
  The default policy may change over time.

  This function only sets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Args:
    device_policy: A device policy.
      Valid values:
      - None: Switch to a system default.
      - 'warn': Copies the tensors which are not on the right device and logs a
        warning.
      - 'explicit': Raises an error if the placement is not as required.
      - 'silent': Silently copies the tensors. Note that this may hide
        performance problems as there is no notification provided when
        operations are blocked on the tensor being copied between devices.
      - 'silent_for_int32': silently copies `int32` tensors, raising errors on
        the other ones.

  Raises:
      ValueError: If an invalid `device_policy` is passed.
  """
    if device_policy == 'silent':
        context.context().device_policy = context.DEVICE_PLACEMENT_SILENT
    elif device_policy == 'silent_for_int32':
        context.context().device_policy = context.DEVICE_PLACEMENT_SILENT_FOR_INT32
    elif device_policy == 'warn':
        context.context().device_policy = context.DEVICE_PLACEMENT_WARN
    elif device_policy == 'explicit':
        context.context().device_policy = context.DEVICE_PLACEMENT_EXPLICIT
    elif device_policy is None:
        context.context().device_policy = None
    else:
        raise ValueError(f'Invalid argument `device_policy`: {device_policy!r}. Please refer to https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_device_policy for valid `device_policy` arguments.')