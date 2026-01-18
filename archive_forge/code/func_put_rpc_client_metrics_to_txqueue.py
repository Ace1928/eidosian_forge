import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def put_rpc_client_metrics_to_txqueue(self, metric_name, action, target, method, call_type, timeout, exception=None):
    kwargs = {'call_type': call_type, 'exchange': target.exchange, 'topic': target.topic, 'namespace': target.namespace, 'version': target.version, 'server': target.server, 'fanout': target.fanout, 'method': method, 'timeout': timeout}
    if exception:
        kwargs['exception'] = exception
    self.put_into_txqueue(metric_name, action, **kwargs)