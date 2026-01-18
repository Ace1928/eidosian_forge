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
def restore_unraisablehook() -> Generator[None, None, None]:
    sys.unraisablehook, prev = (sys.__unraisablehook__, sys.unraisablehook)
    try:
        yield
    finally:
        sys.unraisablehook = prev