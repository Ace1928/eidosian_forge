import logging
import socket
import uuid
import warnings
from array import array
from time import monotonic
from vine import ensure_promise
from . import __version__, sasl, spec
from .abstract_channel import AbstractChannel
from .channel import Channel
from .exceptions import (AMQPDeprecationWarning, ChannelError, ConnectionError,
from .method_framing import frame_handler, frame_writer
from .transport import Transport
def blocking_read(self, timeout=None):
    with self.transport.having_timeout(timeout):
        frame = self.transport.read_frame()
    return self.on_inbound_frame(frame)