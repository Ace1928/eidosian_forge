import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def restore_ops(self, reader=None):
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Args:
      reader: A `CheckpointReader`. If None, a new instance will be created.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    if self._has_registered_saver():
        raise ValueError('Unable to run individual checkpoint restore for objects with registered savers.')
    restore_ops, tensor_saveables, python_positions, _ = self.gather_ops_or_named_saveables()
    restore_ops.extend(self._checkpoint.restore_saveables(tensor_saveables, python_positions, reader=reader))
    return restore_ops