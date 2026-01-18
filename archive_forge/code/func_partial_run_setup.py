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
def partial_run_setup(self, fetches, feeds=None):
    """Sets up the feeds and fetches for partial runs in the session."""
    raise NotImplementedError('partial_run_setup is not implemented for debug-wrapper sessions.')