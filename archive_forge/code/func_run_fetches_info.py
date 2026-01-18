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
def run_fetches_info(self):
    """Get a str representation of the fetches used in the Session.run() call.

    Returns:
      If the information is available from one `Session.run` call, a `str`
        obtained from `repr(fetches)`.
      If the information is available from multiple `Session.run` calls, a
        `list` of `str` from `repr(fetches)`.
      If the information is not available, `None`.
    """
    output = self._run_fetches_info
    return output[0] if len(output) == 1 else output