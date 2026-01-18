from __future__ import annotations
import sys
import time
import warnings
import gevent
from gevent.event import AsyncResult
from gevent.hub import get_hub
import zmq
from zmq import Context as _original_Context
from zmq import Socket as _original_Socket
from .poll import _Poller
def recv_multipart(self, *args, **kwargs):
    """wrap recv_multipart to prevent state_changed on each partial recv"""
    self.__in_recv_multipart = True
    try:
        msg = super().recv_multipart(*args, **kwargs)
    finally:
        self.__in_recv_multipart = False
        self.__state_changed()
    return msg