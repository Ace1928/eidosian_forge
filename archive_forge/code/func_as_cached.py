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
def as_cached(self, key: str, fn: Callable[[], T], ttl: int=None, **kwargs) -> T:
    """
        Caches the return value of a function globally across user sessions, memoizing on the given
        key and supplied keyword arguments.

        Note: Keyword arguments must be hashable.

        Example:

        >>> def load_dataset(name):
        >>>     # some slow operation that uses name to load a dataset....
        >>>     return dataset
        >>> penguins = pn.state.as_cached('dataset-penguins', load_dataset, name='penguins')

        Arguments
        ---------
        key: (str)
          The key to cache the return value under.
        fn: (callable)
          The function or callable whose return value will be cached.
        ttl: (int)
          The number of seconds to keep an item in the cache, or None
          if the cache should not expire. The default is None.
        **kwargs: dict
          Additional keyword arguments to supply to the function,
          which will be memoized over as well.

        Returns
        -------
        Returns the value returned by the cache or the value in
        the cache.
        """
    key = (key,) + tuple(((k, v) for k, v in sorted(kwargs.items())))
    new_expiry = time.monotonic() + ttl if ttl else None
    with self._cache_locks['main']:
        if key in self._cache_locks:
            lock = self._cache_locks[key]
        else:
            self._cache_locks[key] = lock = threading.Lock()
    try:
        with lock:
            if key in self.cache:
                ret, expiry = self.cache.get(key)
            else:
                ret, expiry = (_Undefined, None)
            if ret is _Undefined or (expiry is not None and expiry < time.monotonic()):
                ret, _ = self.cache[key] = (fn(**kwargs), new_expiry)
    finally:
        if not lock.locked() and key in self._cache_locks:
            del self._cache_locks[key]
    return ret