from __future__ import annotations
import os
import sys
from abc import abstractmethod
from asyncio import get_running_loop
from contextlib import contextmanager
from ..utils import SPHINX_AUTODOC_RUNNING
from ctypes import Array, pointer
from ctypes.wintypes import DWORD, HANDLE
from typing import Callable, ContextManager, Iterable, Iterator, TextIO
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.eventloop.win32 import create_win32_event, wait_for_handles
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.win32_types import (
from .ansi_escape_sequences import REVERSE_ANSI_SEQUENCES
from .base import Input
def add_win32_handle(self, handle: HANDLE, callback: Callable[[], None]) -> None:
    """
        Add a Win32 handle to the event loop.
        """
    handle_value = handle.value
    if handle_value is None:
        raise ValueError('Invalid handle.')
    self.remove_win32_handle(handle)
    loop = get_running_loop()
    self._handle_callbacks[handle_value] = callback
    remove_event = create_win32_event()
    self._remove_events[handle_value] = remove_event

    def ready() -> None:
        try:
            callback()
        finally:
            run_in_executor_with_context(wait, loop=loop)

    def wait() -> None:
        result = wait_for_handles([remove_event, handle])
        if result is remove_event:
            windll.kernel32.CloseHandle(remove_event)
            return
        else:
            loop.call_soon_threadsafe(ready)
    run_in_executor_with_context(wait, loop=loop)