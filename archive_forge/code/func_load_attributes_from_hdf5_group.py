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
def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Args:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.

  Returns:
      data: Attributes data.
  """
    if name in group.attrs:
        data = [n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while '%s%d' % (name, chunk_id) in group.attrs:
            data.extend([n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data