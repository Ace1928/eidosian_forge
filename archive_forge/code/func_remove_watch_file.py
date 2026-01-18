from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def remove_watch_file(self, handle: int) -> bool:
    """
        Remove an input file.

        Returns True if the input file exists, False otherwise
        """
    return self._loop.remove_reader(handle)