from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
@property
def checkpointable_objects(self):
    """Returns dictionary of all checkpointable objects."""
    return {key: value for key, value in self._object_dict.items() if value is not None}