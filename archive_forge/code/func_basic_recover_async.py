import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def basic_recover_async(self, requeue=False):
    return self.send_method(spec.Basic.RecoverAsync, 'b', (requeue,))