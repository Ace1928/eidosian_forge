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
def t0(self):
    """Absolute timestamp of the first dumped tensor across all devices.

    Returns:
      (`int`) absolute timestamp of the first dumped tensor, in microseconds.
    """
    return self._t0