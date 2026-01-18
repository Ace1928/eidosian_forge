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
@contextmanager
def pipe_with_overlapped_read() -> Generator[tuple[BufferedWriter, int], None, None]:
    import msvcrt
    from asyncio.windows_utils import pipe
    read_handle, write_handle = pipe(overlapped=(True, False))
    try:
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
        yield (os.fdopen(write_fd, 'wb', closefd=False), read_handle)
    finally:
        kernel32.CloseHandle(Handle(ffi.cast('HANDLE', read_handle)))
        kernel32.CloseHandle(Handle(ffi.cast('HANDLE', write_handle)))