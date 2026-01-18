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
def saveable_object_to_restore_fn(saveables):
    """Generates `Trackable._restore_from_tensors` from SaveableObjects."""

    def _restore_from_tensors(restored_tensors):
        restore_ops = {}
        for saveable in saveables:
            saveable_restored_tensors = []
            for spec in saveable.specs:
                name = trackable_utils.extract_local_name(_convert_to_string(spec.name))
                slice_spec = _convert_to_string(spec.slice_spec)
                maybe_tensor = restored_tensors[name]
                if not isinstance(maybe_tensor, dict):
                    maybe_tensor = {'': maybe_tensor}
                saveable_restored_tensors.append(maybe_tensor[slice_spec])
            restore_ops[saveable.name] = saveable.restore(saveable_restored_tensors, restored_shapes=None)
        return restore_ops
    return _restore_from_tensors