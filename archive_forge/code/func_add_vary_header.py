from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def add_vary_header(self, vary: str) -> None:
    existing = self.get('vary')
    if existing is not None:
        vary = ', '.join([existing, vary])
    self['vary'] = vary