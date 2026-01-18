from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop
def final_handler(signal_number: int):
    handler(signal_number, None)
    return GLib.SOURCE_CONTINUE