from __future__ import annotations
import socket
from contextlib import contextmanager
from functools import partial
from itertools import count
from time import sleep
from .common import ignore_errors
from .log import get_logger
from .messaging import Consumer, Producer
from .utils.compat import nested
from .utils.encoding import safe_repr
from .utils.limits import TokenBucket
from .utils.objects import cached_property
@property
def producer_connection(self):
    if self._producer_connection is None:
        conn = self.connection.clone()
        conn.ensure_connection(self.on_connection_error, self.connect_max_retries)
        self._producer_connection = conn
    return self._producer_connection