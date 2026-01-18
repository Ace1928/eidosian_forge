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
def lame_workers(self):
    """Ping all workers, returning manager containing lame workers (or None)."""
    ping_results = self.ping()
    lame_workers = []
    for ping_response, device, op in zip(ping_results, self._devices, self._ops):
        if ping_response.health_status != event_pb2.OK:
            lame_workers.append((device, op))
    if not lame_workers:
        return None
    bad_devices, bad_ops = zip(*lame_workers)
    return WorkerHeartbeatManager(self._session, bad_devices, bad_ops, self._request_placeholder)