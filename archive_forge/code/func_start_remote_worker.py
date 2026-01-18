import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
def start_remote_worker(self, worker_tags=None):
    """Runs a tf.data service worker in a remote process."""
    pipe_reader, pipe_writer = multi_process_lib.multiprocessing.Pipe(duplex=False)
    worker_process = _RemoteWorkerProcess(self.dispatcher_address(), port=test_util.pick_unused_port(), worker_tags=worker_tags, pipe_writer=pipe_writer)
    worker_process.start()
    worker_address = pipe_reader.recv()
    self._remote_workers.append((worker_address, worker_process))