from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
@property
def quantization_mode(self):
    """The quantization mode of this policy.

        Returns:
            The quantization mode of this policy, as a string.
        """
    return self._quantization_mode