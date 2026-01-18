from __future__ import absolute_import
import time
import msvcrt  # pylint: disable=import-error
import contextlib
from jinxed import win32  # pylint: disable=import-error
from .terminal import WINSZ
from .terminal import Terminal as _Terminal
def kbhit(self, timeout=None):
    """
        Return whether a keypress has been detected on the keyboard.

        This method is used by :meth:`inkey` to determine if a byte may
        be read using :meth:`getch` without blocking.  This is implemented
        by wrapping msvcrt.kbhit() in a timeout.

        :arg float timeout: When ``timeout`` is 0, this call is
            non-blocking, otherwise blocking indefinitely until keypress
            is detected when None (default). When ``timeout`` is a
            positive number, returns after ``timeout`` seconds have
            elapsed (float).
        :rtype: bool
        :returns: True if a keypress is awaiting to be read on the keyboard
            attached to this terminal.
        """
    end = time.time() + (timeout or 0)
    while True:
        if msvcrt.kbhit():
            return True
        if timeout is not None and end < time.time():
            break
        time.sleep(0.01)
    return False