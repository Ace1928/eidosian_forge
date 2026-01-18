from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def multi_items(self) -> list[tuple[_KeyType, _CovariantValueType]]:
    return list(self._list)