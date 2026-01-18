import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
def set_tpu_core_ids(self, mesh_name, tpu_core_ids):
    """Sets the singleton global device ID-to-physical core ID map.

    Args:
      mesh_name: The name of a mesh. If empty, set the default mapping.
      tpu_core_ids: TPU core IDs sorted by TF task/device ordinal.
    """
    _pywrap_dtensor_device.SetTPUCoreIDs(self._device_info, mesh_name, tpu_core_ids)