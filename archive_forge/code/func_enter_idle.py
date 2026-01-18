from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def enter_idle(self, callback: Callable[[], typing.Any]) -> int:
    """
        Add a callback for entering idle.

        Returns a handle that may be passed to remove_enter_idle()
        """
    self._idle_handle += 1
    self._idle_callbacks[self._idle_handle] = callback
    return self._idle_handle