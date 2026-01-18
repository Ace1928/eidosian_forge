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
def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
    """Saves optimizer weights of a optimizer to a HDF5 group.

  Args:
      hdf5_group: HDF5 group.
      optimizer: optimizer instance.
  """
    symbolic_weights = getattr(optimizer, 'weights')
    if symbolic_weights:
        weights_group = hdf5_group.create_group('optimizer_weights')
        weight_names = [str(w.name).encode('utf8') for w in symbolic_weights]
        save_attributes_to_hdf5_group(weights_group, 'weight_names', weight_names)
        weight_values = backend.batch_get_value(symbolic_weights)
        for name, val in zip(weight_names, weight_values):
            param_dset = weights_group.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val