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
def save_attributes_to_hdf5_group(group, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.

  This method deals with an inherent problem of HDF5 file which is not
  able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Args:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to save.
      data: Attributes data to store.

  Raises:
    RuntimeError: If any single attribute is too large to be saved.
  """
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]
    if bad_attributes:
        raise RuntimeError('The following attributes cannot be saved to HDF5 file because they are larger than %d bytes: %s' % (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))
    data_npy = np.asarray(data)
    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)
    while any((x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data)):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)
    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs['%s%d' % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data