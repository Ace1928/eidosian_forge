import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
def load_model_from_hdf5(filepath, custom_objects=None, compile=True):
    """Loads a model saved via `save_model_to_hdf5`.

  Args:
      filepath: One of the following:
          - String, path to the saved model
          - `h5py.File` object from which to load the model
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
      compile: Boolean, whether to compile the model
          after loading.

  Returns:
      A Keras model instance. If an optimizer was found
      as part of the saved model, the model is already
      compiled. Otherwise, the model is uncompiled and
      a warning will be displayed. When `compile` is set
      to False, the compilation is omitted without any
      warning.

  Raises:
      ImportError: if h5py is not available.
      ValueError: In case of an invalid savefile.
  """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    if not custom_objects:
        custom_objects = {}
    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    model = None
    try:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        if hasattr(model_config, 'decode'):
            model_config = model_config.decode('utf-8')
        model_config = json_utils.decode(model_config)
        model = model_config_lib.model_from_config(model_config, custom_objects=custom_objects)
        load_weights_from_hdf5_group(f['model_weights'], model.layers)
        if compile:
            training_config = f.attrs.get('training_config')
            if hasattr(training_config, 'decode'):
                training_config = training_config.decode('utf-8')
            if training_config is None:
                logging.warning('No training configuration found in the save file, so the model was *not* compiled. Compile it manually.')
                return model
            training_config = json_utils.decode(training_config)
            model.compile(**saving_utils.compile_args_from_training_config(training_config, custom_objects), from_serialized=True)
            saving_utils.try_build_compiled_arguments(model)
            if 'optimizer_weights' in f:
                try:
                    model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError):
                    logging.warning('Error when creating the weights of optimizer {}, making it impossible to restore the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.')
                optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning('Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.')
    finally:
        if opened_new_file:
            f.close()
    return model