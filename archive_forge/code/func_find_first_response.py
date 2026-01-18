from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
def find_first_response(cache: DiskCache, urls: Sequence[str], cache_fetch_timeout: float | int | None=None, session: requests.Session | None=None) -> str:
    """Decode the first successfully fetched URL, from UTF-8 encoding to Python unicode."""
    session_created = False
    if session is None:
        session = requests.Session()
        session.mount('file://', FileAdapter())
        session_created = True
    try:
        for url in urls:
            try:
                return cache.cached_fetch_url(session=session, url=url, timeout=cache_fetch_timeout)
            except requests.exceptions.RequestException:
                LOG.exception('Exception reading Public Suffix List url %s', url)
    finally:
        if session_created:
            session.close()
    raise SuffixListNotFound('No remote Public Suffix List found. Consider using a mirror, or avoid this fetch by constructing your TLDExtract with `suffix_list_urls=()`.')