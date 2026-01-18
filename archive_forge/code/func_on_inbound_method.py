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
def on_inbound_method(self, channel_id, method_sig, payload, content):
    if self.channels is None:
        raise RecoverableConnectionError('Connection already closed')
    return self.channels[channel_id].dispatch_method(method_sig, payload, content)