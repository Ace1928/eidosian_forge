import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def put_rpc_server_metrics_to_txqueue(self, metric_name, action, target, endpoint, ns, ver, method, exception=None):
    kwargs = {'endpoint': endpoint, 'namespace': ns, 'version': ver, 'method': method, 'exchange': None, 'topic': None, 'server': None}
    if target:
        kwargs['exchange'] = target.exchange
        kwargs['topic'] = target.topic
        kwargs['server'] = target.server
    if exception:
        kwargs['exception'] = exception
    self.put_into_txqueue(metric_name, action, **kwargs)