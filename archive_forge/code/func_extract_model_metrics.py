import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def extract_model_metrics(model):
    """Convert metrics from a Keras model `compile` API to dictionary.

  This is used for converting Keras models to Estimators and SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to metric instances. May return `None` if
    the model does not contain any metrics.
  """
    if getattr(model, '_compile_metrics', None):
        return {m.name: m for m in model._compile_metric_functions}
    return None