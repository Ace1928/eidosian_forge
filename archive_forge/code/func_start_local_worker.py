import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
def start_local_worker(self, worker_tags=None):
    worker = data_service_test_base.TestWorker(self.dispatcher_address(), _WORKER_SHUTDOWN_QUIET_PERIOD_MS, port=test_util.pick_unused_port(), worker_tags=worker_tags)
    worker.start()
    self._local_workers.append(worker)