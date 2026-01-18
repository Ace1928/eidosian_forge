from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
def set_rtt(rtt: float) -> None:
    RTT.set(rtt)