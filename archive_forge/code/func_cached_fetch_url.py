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
def cached_fetch_url(self, session: requests.Session, url: str, timeout: float | int | None) -> str:
    """Get a url but cache the response."""
    return self.run_and_cache(func=_fetch_url, namespace='urls', kwargs={'session': session, 'url': url, 'timeout': timeout}, hashed_argnames=['url'])