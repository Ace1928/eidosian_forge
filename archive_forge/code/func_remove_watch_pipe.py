from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def remove_watch_pipe(self, write_fd: int) -> bool:
    """
            Close the read end of the pipe and remove the watch created by :meth:`watch_pipe`.

            ..note:: You are responsible for closing the write end of the pipe.

            Returns ``True`` if the watch pipe exists, ``False`` otherwise
            """
    try:
        watch_handle, pipe_rd = self._watch_pipes.pop(write_fd)
    except KeyError:
        return False
    if not self.event_loop.remove_watch_file(watch_handle):
        return False
    os.close(pipe_rd)
    return True