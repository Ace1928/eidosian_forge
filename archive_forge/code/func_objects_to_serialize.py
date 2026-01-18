import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
@abc.abstractmethod
def objects_to_serialize(self, serialization_cache):
    """Returns dictionary of extra checkpointable objects to serialize.

    See `functions_to_serialize` for an explanation of this function's
    effects.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to checkpointable objects.
    """
    raise NotImplementedError