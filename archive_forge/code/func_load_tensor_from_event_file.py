import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def load_tensor_from_event_file(event_file_path):
    """Load a tensor from an event file.

  Assumes that the event file contains a `Event` protobuf and the `Event`
  protobuf contains a `Tensor` value.

  Args:
    event_file_path: (`str`) path to the event file.

  Returns:
    The tensor value loaded from the event file, as a `numpy.ndarray`. For
    uninitialized Tensors, returns `None`. For Tensors of data types that
    cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return
    `None`.
  """
    event = event_pb2.Event()
    with gfile.Open(event_file_path, 'rb') as f:
        event.ParseFromString(f.read())
        return load_tensor_from_event(event)