from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def in_place_subclassed_model_state_restoration(model):
    """Restores the original state of a model after it was "reset".

  This undoes this action of `_in_place_subclassed_model_reset`, which is called
  in `clone_and_build_model` if `in_place_reset` is set to True.

  Args:
    model: Instance of a Keras model created via subclassing, on which
      `_in_place_subclassed_model_reset` was previously called.
  """
    assert not model._is_graph_network
    if hasattr(model, '_original_attributes_cache') and model._original_attributes_cache is not None:
        setattr_tracking = model._setattr_tracking
        model._setattr_tracking = False
        model._self_tracked_trackables = []
        for name, value in model._original_attributes_cache.items():
            setattr(model, name, value)
            if isinstance(value, Layer):
                model._self_tracked_trackables.append(value)
        model._original_attributes_cache = None
        model._setattr_tracking = setattr_tracking
    else:
        _reset_build_compile_trackers(model)