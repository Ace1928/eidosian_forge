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
@contextmanager
def extra_context(self, connection, channel):
    yield