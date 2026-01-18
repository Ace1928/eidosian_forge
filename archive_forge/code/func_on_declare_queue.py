import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def on_declare_queue(queue):
    new_queue = queue(producer.channel)
    new_queue.name = queues.get(queue.name, queue.name)
    if new_queue.routing_key == queue.name:
        new_queue.routing_key = queues.get(queue.name, new_queue.routing_key)
    if new_queue.exchange.name == queue.name:
        new_queue.exchange.name = queues.get(queue.name, queue.name)
    new_queue.declare()