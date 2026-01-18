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
def onload(self, callback: Callable[[], None | Awaitable[None]] | Coroutine[Any, Any, None], threaded: bool=False):
    """
        Callback that is triggered when a session has been served.

        Arguments
        ---------
        callback: Callable[[], None] | Coroutine[Any, Any, None]
           Callback that is executed when the application is loaded
        threaded: bool
          Whether the onload callback can be threaded
        """
    if self.curdoc is None or self._is_pyodide or self.loaded:
        if self._thread_pool:
            self.execute(callback, schedule='threaded')
        else:
            self.execute(callback, schedule=False)
        return
    elif self.curdoc not in self._onload:
        self._onload[self.curdoc] = []
    self._onload[self.curdoc].append((callback, threaded))