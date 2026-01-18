import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def basic_publish_confirm(self, *args, **kwargs):
    confirm_timeout = kwargs.pop('confirm_timeout', None)

    def confirm_handler(method, *args):
        if method == spec.Basic.Nack:
            raise MessageNacked()
    if not self._confirm_selected:
        self._confirm_selected = True
        self.confirm_select()
    ret = self._basic_publish(*args, **kwargs)
    timeout = confirm_timeout or kwargs.get('timeout', None)
    self.wait([spec.Basic.Ack, spec.Basic.Nack], callback=confirm_handler, timeout=timeout)
    return ret