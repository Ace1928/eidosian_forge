from __future__ import annotations
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from functools import partial
from itertools import count
from uuid import NAMESPACE_OID, uuid3, uuid4, uuid5
from amqp import ChannelError, RecoverableConnectionError
from .entity import Exchange, Queue
from .log import get_logger
from .serialization import registry as serializers
from .utils.uuid import uuid
def drain_consumer(consumer, limit=1, timeout=None, callbacks=None):
    """Drain messages from consumer instance."""
    acc = deque()

    def on_message(body, message):
        acc.append((body, message))
    consumer.callbacks = [on_message] + (callbacks or [])
    with consumer:
        for _ in eventloop(consumer.channel.connection.client, limit=limit, timeout=timeout, ignore_timeouts=True):
            try:
                yield acc.popleft()
            except IndexError:
                pass