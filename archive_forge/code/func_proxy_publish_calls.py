import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
def proxy_publish_calls(self, calls, routing_key, exc_type=mock.ANY):
    return [mock.call.connection.Producer(), mock.call.connection.Producer().__enter__(), mock.call.connection.ensure(mock.ANY, mock.ANY, interval_start=mock.ANY, interval_max=mock.ANY, max_retries=mock.ANY, interval_step=mock.ANY, errback=mock.ANY), mock.call.Queue(name=self._queue_name(routing_key), routing_key=routing_key, exchange=self.exchange_inst_mock, durable=False, auto_delete=True, channel=None)] + calls + [mock.call.connection.Producer().__exit__(exc_type, mock.ANY, mock.ANY)]