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
@property
def python_graph(self):
    """Get the Python graph.

    Returns:
      If the Python graph has been set, returns a `tf.Graph` object. Otherwise,
      returns None.
    """
    return self._python_graph