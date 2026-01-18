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
def reconstructed_non_debug_partition_graphs(self):
    """Reconstruct partition graphs with the debugger-inserted ops stripped.

    The reconstructed partition graphs are identical to the original (i.e.,
    non-debugger-decorated) partition graphs except in the following respects:
      1) The exact names of the runtime-inserted internal nodes may differ.
         These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.
      2) As a consequence of 1, the nodes that receive input directly from such
         send- and recv-type ops will have different input names.
      3) The parallel_iteration attribute of while-loop Enter ops are set to 1.

    Returns:
      A dict mapping device names (`str`s) to reconstructed
      `tf.compat.v1.GraphDef`s.
    """
    non_debug_graphs = {}
    for key in self._debug_graphs:
        non_debug_graphs[key] = self._debug_graphs[key].non_debug_graph_def
    return non_debug_graphs