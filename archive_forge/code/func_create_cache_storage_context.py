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
def create_cache_storage_context(self, function_key: str, function_name: str, persist: CachePersistType, ttl_seconds: float | None, max_entries: int | None) -> CacheStorageContext:
    return CacheStorageContext(function_key=function_key, function_display_name=function_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)