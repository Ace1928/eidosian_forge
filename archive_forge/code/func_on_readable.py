from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def on_readable(self, fd, _pycurl=pycurl):
    return self._on_event(fd, _pycurl.CSELECT_IN)