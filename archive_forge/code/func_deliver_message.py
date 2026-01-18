import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
def deliver_message(self, topic, ctxt, message, server=None, fanout=False, reply_q=None):
    with self._queues_lock:
        if fanout:
            queues = [q for t, q in self._server_queues.items() if t[0] == topic]
        elif server is not None:
            queues = [self._get_server_queue(topic, server)]
        else:
            self._get_topic_queue(topic)
            queues = [q for t, q in self._topic_queues.items() if t[0] == topic]

        def requeue():
            self.deliver_message(topic, ctxt, message, server=server, fanout=fanout, reply_q=reply_q)
        for q in queues:
            q.append((ctxt, message, reply_q, requeue))