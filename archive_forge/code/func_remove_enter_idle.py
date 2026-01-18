from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def remove_enter_idle(self, handle: int) -> bool:
    """
        Remove an idle callback.

        Returns True if the handle was removed.
        """
    try:
        del self._idle_callbacks[handle]
    except KeyError:
        return False
    return True