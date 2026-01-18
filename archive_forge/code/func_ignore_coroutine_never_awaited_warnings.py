from __future__ import annotations
import asyncio
import gc
import os
import socket as stdlib_socket
import sys
import warnings
from contextlib import closing, contextmanager
from typing import TYPE_CHECKING, TypeVar
import pytest
from trio._tests.pytest_plugin import RUN_SLOW
@contextmanager
def ignore_coroutine_never_awaited_warnings() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="coroutine '.*' was never awaited")
        try:
            yield
        finally:
            gc_collect_harder()