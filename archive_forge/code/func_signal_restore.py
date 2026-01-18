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
def signal_restore(self) -> None:
    """
        Called in the finally block of run wrapper to restore the
        SIGTSTP, SIGCONT and SIGWINCH signal handlers.

        Override this function to call from main thread in threaded
        applications.
        """
    self.signal_handler_setter(signal.SIGTSTP, self._prev_sigtstp_handler or signal.SIG_DFL)
    self.signal_handler_setter(signal.SIGCONT, self._prev_sigcont_handler or signal.SIG_DFL)
    self.signal_handler_setter(signal.SIGWINCH, self._prev_sigwinch_handler or signal.SIG_DFL)