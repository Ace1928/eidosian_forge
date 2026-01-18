import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
def ensure_queue(self, target, pool):
    with self._queues_lock:
        if target.server:
            self._get_server_queue(target.topic, target.server)
        else:
            self._get_topic_queue(target.topic, pool)