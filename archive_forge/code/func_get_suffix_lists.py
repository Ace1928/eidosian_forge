from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
def get_suffix_lists(cache: DiskCache, urls: Sequence[str], cache_fetch_timeout: float | int | None, fallback_to_snapshot: bool, session: requests.Session | None=None) -> tuple[list[str], list[str]]:
    """Fetch, parse, and cache the suffix lists."""
    return cache.run_and_cache(func=_get_suffix_lists, namespace='publicsuffix.org-tlds', kwargs={'cache': cache, 'urls': urls, 'cache_fetch_timeout': cache_fetch_timeout, 'fallback_to_snapshot': fallback_to_snapshot, 'session': session}, hashed_argnames=['urls', 'fallback_to_snapshot'])