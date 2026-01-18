from __future__ import annotations
import pickle
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, Literal, TypeVar, Union, cast, overload
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import CacheError, CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.caching.storage import (
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.dummy_cache_storage import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
def read_result(self, key: str) -> CachedResult:
    """Read a value and messages from the cache. Raise `CacheKeyNotFoundError`
        if the value doesn't exist, and `CacheError` if the value exists but can't
        be unpickled.
        """
    try:
        pickled_entry = self.storage.get(key)
    except CacheStorageKeyNotFoundError as e:
        raise CacheKeyNotFoundError(str(e)) from e
    except CacheStorageError as e:
        raise CacheError(str(e)) from e
    try:
        entry = pickle.loads(pickled_entry)
        if not isinstance(entry, MultiCacheResults):
            self.storage.delete(key)
            raise CacheKeyNotFoundError()
        ctx = get_script_run_ctx()
        if not ctx:
            raise CacheKeyNotFoundError()
        widget_key = entry.get_current_widget_key(ctx, CacheType.DATA)
        if widget_key in entry.results:
            return entry.results[widget_key]
        else:
            raise CacheKeyNotFoundError()
    except pickle.UnpicklingError as exc:
        raise CacheError(f'Failed to unpickle {key}') from exc