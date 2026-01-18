import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def run_server(self, blocking=True):
    """Start running the server.

    Args:
      blocking: If `True`, block until `stop_server()` is invoked.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has already started running.
    """
    self._server_lock.acquire()
    try:
        if self._stop_requested:
            raise ValueError('Server has already stopped')
        if self._server_started:
            raise ValueError('Server has already started running')
        no_max_message_sizes = [('grpc.max_receive_message_length', -1), ('grpc.max_send_message_length', -1)]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=no_max_message_sizes)
        debug_service_pb2_grpc.add_EventListenerServicer_to_server(self, self.server)
        self.server.add_insecure_port('[::]:%d' % self._server_port)
        self.server.start()
        self._server_started = True
    finally:
        self._server_lock.release()
    if blocking:
        while not self._stop_requested:
            time.sleep(1.0)