import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def rpc_server_processing_seconds(self, target, endpoint, ns, ver, method, duration):
    self.put_rpc_server_metrics_to_txqueue('rpc_server_processing_seconds', message_type.MetricAction('observe', duration), target, endpoint, ns, ver, method)