from __future__ import annotations
import errno
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import (
import requests
from filelock import FileLock
def run_and_cache(self, func: Callable[..., T], namespace: str, kwargs: dict[str, Hashable], hashed_argnames: Iterable[str]) -> T:
    """Get a url but cache the response."""
    if not self.enabled:
        return func(**kwargs)
    key_args = {k: v for k, v in kwargs.items() if k in hashed_argnames}
    cache_filepath = self._key_to_cachefile_path(namespace, key_args)
    lock_path = cache_filepath + '.lock'
    try:
        _make_dir(cache_filepath)
    except OSError as ioe:
        global _DID_LOG_UNABLE_TO_CACHE
        if not _DID_LOG_UNABLE_TO_CACHE:
            LOG.warning('unable to cache %s.%s in %s. This could refresh the Public Suffix List over HTTP every app startup. Construct your `TLDExtract` with a writable `cache_dir` or set `cache_dir=None` to silence this warning. %s', namespace, key_args, cache_filepath, ioe)
            _DID_LOG_UNABLE_TO_CACHE = True
        return func(**kwargs)
    with FileLock(lock_path, timeout=self.lock_timeout):
        try:
            result = cast(T, self.get(namespace=namespace, key=key_args))
        except KeyError:
            result = func(**kwargs)
            self.set(namespace=namespace, key=key_args, value=result)
        return result