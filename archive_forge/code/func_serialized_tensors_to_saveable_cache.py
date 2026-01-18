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
def serialized_tensors_to_saveable_cache(serialized_tensors):
    """Converts a tensor dict to a SaveableObject cache.

  Args:
    serialized_tensors: Map from Trackable to a tensor dict. The tensor dict
      maps checkpoint key (-> slice_spec) -> Tensor

  Returns:
    A dict mapping Trackable objects to a map from local savable name to
    SaveableObject.
  """
    saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()
    for obj, tensor_dict in serialized_tensors.items():
        if not tensor_dict:
            continue
        if isinstance(obj, SaveableCompatibilityConverter):
            trackable_obj = obj.obj
            saveables_cache[trackable_obj] = {}
            for saveable in obj.saveables:
                local_name = trackable_utils.extract_local_name(saveable.name)
                saveables_cache[trackable_obj][local_name] = [saveable]
            continue
        specs = []
        local_names = []
        prefix = saveable_compat.get_saveable_name(obj) or ''
        for checkpoint_key, maybe_tensor in tensor_dict.items():
            if not isinstance(maybe_tensor, dict):
                maybe_tensor = {'': maybe_tensor}
            for slice_spec, tensor in maybe_tensor.items():
                if isinstance(tensor, saveable_object.SaveSpec):
                    specs.append(tensor)
                else:
                    specs.append(saveable_object.SaveSpec(tensor, slice_spec, checkpoint_key))
            local_names.append(trackable_utils.extract_local_name(checkpoint_key, prefix))
        object_name = trackable_utils.extract_object_name(next(iter(tensor_dict.keys())))
        saveables_cache[obj] = {trackable_utils.SERIALIZE_TO_TENSORS_NAME: [TrackableSaveable(obj, specs, object_name, local_names=local_names, prefix=prefix)]}
    return saveables_cache