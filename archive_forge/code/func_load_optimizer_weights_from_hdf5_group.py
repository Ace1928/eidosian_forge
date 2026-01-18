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
def load_optimizer_weights_from_hdf5_group(hdf5_group):
    """Load optimizer weights from a HDF5 group.

  Args:
      hdf5_group: A pointer to a HDF5 group.

  Returns:
      data: List of optimizer weight names.
  """
    weights_group = hdf5_group['optimizer_weights']
    optimizer_weight_names = load_attributes_from_hdf5_group(weights_group, 'weight_names')
    return [weights_group[weight_name] for weight_name in optimizer_weight_names]