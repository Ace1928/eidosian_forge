from __future__ import annotations
import contextlib
import fcntl
import functools
import os
import selectors
import signal
import struct
import sys
import termios
import tty
import typing
from subprocess import PIPE, Popen
from urwid import signals
from . import _raw_display_base, escape
from .common import INPUT_DESCRIPTORS_CHANGED
def signal_init(self) -> None:
    """
        Called in the startup of run wrapper to set the SIGWINCH
        and SIGTSTP signal handlers.

        Override this function to call from main thread in threaded
        applications.
        """
    self._prev_sigwinch_handler = self.signal_handler_setter(signal.SIGWINCH, self._sigwinch_handler)
    self._prev_sigtstp_handler = self.signal_handler_setter(signal.SIGTSTP, self._sigtstp_handler)