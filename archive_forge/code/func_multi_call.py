from __future__ import annotations
import socket
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import copy
from itertools import count
from time import time
from . import Consumer, Exchange, Producer, Queue
from .clocks import LamportClock
from .common import maybe_declare, oid_from
from .exceptions import InconsistencyError
from .log import get_logger
from .matcher import match
from .utils.functional import maybe_evaluate, reprcall
from .utils.objects import cached_property
from .utils.uuid import uuid
def multi_call(self, command, kwargs=None, timeout=1, limit=None, callback=None, channel=None):
    kwargs = {} if not kwargs else kwargs
    return self._broadcast(command, kwargs, reply=True, timeout=timeout, limit=limit, callback=callback, channel=channel)