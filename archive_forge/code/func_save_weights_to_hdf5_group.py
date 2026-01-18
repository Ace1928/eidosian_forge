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
def save_weights_to_hdf5_group(f, layers):
    """Saves the weights of a list of layers to a HDF5 group.

  Args:
      f: HDF5 group.
      layers: List of layer instances.
  """
    from tensorflow.python.keras import __version__ as keras_version
    save_attributes_to_hdf5_group(f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
    f.attrs['backend'] = backend.backend().encode('utf8')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')
    for layer in sorted(layers, key=lambda x: x.name):
        g = f.create_group(layer.name)
        weights = _legacy_weights(layer)
        weight_values = backend.batch_get_value(weights)
        weight_names = [w.name.encode('utf8') for w in weights]
        save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
        for name, val in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val