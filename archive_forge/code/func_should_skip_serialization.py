import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def should_skip_serialization(layer):
    """Skip serializing extra objects and functions if layer inputs aren't set."""
    saved_model_input_spec_set = isinstance(layer, training_lib.Model) and layer._saved_model_inputs_spec is not None
    if not layer.built and (not saved_model_input_spec_set):
        logging.warning('Skipping full serialization of Keras layer {}, because it is not built.'.format(layer))
        return True
    return False