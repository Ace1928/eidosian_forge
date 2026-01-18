import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def is_empty(x):
    """Check whether a possibly nested structure is empty."""
    if not nest.is_nested(x):
        return False
    if isinstance(x, collections_abc.Mapping):
        return is_empty(list(x.values()))
    for item in x:
        if not is_empty(item):
            return False
    return True