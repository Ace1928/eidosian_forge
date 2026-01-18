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
@cached_property
def reply_queue(self):
    return self.get_reply_queue()