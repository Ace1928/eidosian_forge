from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
@property
def object_identifier(self):
    return constants.METRIC_IDENTIFIER