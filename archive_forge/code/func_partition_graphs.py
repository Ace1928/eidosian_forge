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
def partition_graphs(self):
    """Get the partition graphs.

    Returns:
      Partition graphs as a list of GraphDef.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """
    if not self._debug_graphs:
        raise LookupError('No partition graphs have been loaded.')
    return [self._debug_graphs[key].debug_graph_def for key in self._debug_graphs]