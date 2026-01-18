import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
def get_current_worker_context():
    """Returns the current task context."""
    try:
        return _worker_context.current
    except AttributeError:
        return None