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
def collect_replies(conn, channel, queue, *args, **kwargs):
    """Generator collecting replies from ``queue``."""
    no_ack = kwargs.setdefault('no_ack', True)
    received = False
    try:
        for body, message in itermessages(conn, channel, queue, *args, **kwargs):
            if not no_ack:
                message.ack()
            received = True
            yield body
    finally:
        if received:
            channel.after_reply_message_received(queue.name)