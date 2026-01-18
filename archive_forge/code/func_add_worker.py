import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def add_worker(self, start=True):
    worker = TestWorker(self.dispatcher_address(), self._worker_shutdown_quiet_period_ms, self._protocol, self._data_transfer_protocol, snapshot_max_chunk_size_bytes=self._snapshot_max_chunk_size_bytes)
    if start:
        worker.start()
    self.workers.append(worker)