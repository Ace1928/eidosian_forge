from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
def convert_to_trackable(obj, parent=None):
    """Converts `obj` to `Trackable`."""
    if isinstance(obj, base.Trackable):
        return obj
    obj = data_structures.wrap_or_unwrap(obj)
    if tensor_util.is_tf_type(obj) and obj.dtype not in (dtypes.variant, dtypes.resource) and (not resource_variable_ops.is_resource_variable(obj)):
        return saved_model_utils.TrackableConstant(obj, parent)
    if not isinstance(obj, base.Trackable):
        raise ValueError(f'Cannot convert {obj} to Trackable.')
    return obj