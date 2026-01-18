from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def dgen():
    if isinstance(at, Iterator):
        while True:
            new = next(at)
            yield new.timestamp()
    elif callable(at):
        while True:
            new = at(dt.datetime.utcnow())
            if new is None:
                raise StopIteration
            yield new.replace(tzinfo=dt.timezone.utc).astimezone().timestamp()
    elif period is None:
        yield at.timestamp()
        raise StopIteration
    new_time = at or dt.datetime.now()
    while True:
        yield new_time.timestamp()
        new_time += period