import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def create_saveable_object(name, key, factory, call_with_mapped_captures):
    """Creates a SaveableObject while potentially in a different graph.

  When creating the frozen saver for SavedModel, the save and restore ops are
  placed in a separate graph. Since RestoredSaveableObject uses tf.functions to
  save and restore, the function captures must be mapped to the new graph.

  Args:
    name: Name of SaveableObject factory.
    key: Checkpoint key of this SaveableObject.
    factory: Factory method for creating the SaveableObject.
    call_with_mapped_captures: Helper that calls a tf.function while remapping
      the captures.

  Returns:
    a SaveableObject.
  """
    if call_with_mapped_captures is None:
        return factory(name=key)
    if name == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
        return factory(name=key, call_with_mapped_captures=call_with_mapped_captures)
    elif is_factory_for_restored_saveable_object(factory):
        concrete_save_fn = factory.keywords['save_function']

        def save_fn(name):
            return call_with_mapped_captures(concrete_save_fn, [name])
        concrete_restore_fn = factory.keywords['restore_function']

        def restore_fn(*restored_tensors):
            return call_with_mapped_captures(concrete_restore_fn, restored_tensors)
        return factory(save_function=save_fn, restore_function=restore_fn, name=key)
    else:
        return factory(name=key)