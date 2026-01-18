from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
def compute_value_lock(self, value_key: str) -> threading.Lock:
    """Return the lock that should be held while computing a new cached value.
        In a popular app with a cache that hasn't been pre-warmed, many sessions may try
        to access a not-yet-cached value simultaneously. We use a lock to ensure that
        only one of those sessions computes the value, and the others block until
        the value is computed.
        """
    with self._value_locks_lock:
        return self._value_locks[value_key]