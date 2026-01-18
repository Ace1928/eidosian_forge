import threading
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
def all_worker_devices(session):
    """Return a list of devices for each worker in the system."""
    devices = session.list_devices()
    devices_that_support_heartbeats = []
    for device in devices:
        name = device.name
        if ':TPU:0' in name and 'coordinator' not in name:
            devices_that_support_heartbeats.append(name.replace('TPU', 'CPU'))
    return devices_that_support_heartbeats