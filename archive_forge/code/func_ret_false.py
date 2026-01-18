from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop
@self.handle_exit
def ret_false() -> Literal[False]:
    callback()
    self._enable_glib_idle()
    return False