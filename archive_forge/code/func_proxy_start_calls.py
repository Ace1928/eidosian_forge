import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
def proxy_start_calls(self, calls, exc_type=mock.ANY):
    return [mock.call.Queue(name=self._queue_name(self.topic), exchange=self.exchange_inst_mock, routing_key=self.topic, durable=False, auto_delete=True, channel=self.conn_inst_mock), mock.call.connection.Consumer(queues=self.queue_inst_mock, callbacks=[mock.ANY]), mock.call.connection.Consumer().__enter__(), mock.call.connection.ensure(mock.ANY, mock.ANY, interval_start=mock.ANY, interval_max=mock.ANY, max_retries=mock.ANY, interval_step=mock.ANY, errback=mock.ANY)] + calls + [mock.call.connection.Consumer().__exit__(exc_type, mock.ANY, mock.ANY)]