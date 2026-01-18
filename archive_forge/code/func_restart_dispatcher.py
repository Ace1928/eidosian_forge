import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
def restart_dispatcher(self):
    port = int(self.dispatcher_address().split(':')[1])
    self._dispatcher._stop()
    self._start_dispatcher(worker_addresses=self.local_worker_addresses() + self.remote_worker_addresses(), port=port)