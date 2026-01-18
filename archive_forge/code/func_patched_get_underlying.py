from __future__ import annotations
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import create_autospec
import pytest
from ... import _core, sleep
from ...testing import wait_all_tasks_blocked
from .tutil import gc_collect_harder, restore_unraisablehook, slow
def patched_get_underlying(sock: int | CData, *, which: int=WSAIoctls.SIO_BASE_HANDLE) -> CData:
    if hasattr(sock, 'fileno'):
        sock = sock.fileno()
    if which == WSAIoctls.SIO_BASE_HANDLE:
        raise OSError('nope')
    else:
        return _handle(sock)