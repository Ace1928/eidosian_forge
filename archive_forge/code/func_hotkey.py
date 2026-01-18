from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
@_genericPyAutoGUIChecks
def hotkey(*args, **kwargs):
    """Performs key down presses on the arguments passed in order, then performs
    key releases in reverse order.

    The effect is that calling hotkey('ctrl', 'shift', 'c') would perform a
    "Ctrl-Shift-C" hotkey/keyboard shortcut press.

    Args:
      key(s) (str): The series of keys to press, in order. This can also be a
        list of key strings to press.
      interval (float, optional): The number of seconds in between each press.
        0.0 by default, for no pause in between presses.

    Returns:
      None
    """
    interval = float(kwargs.get('interval', 0.0))
    if len(args) and isinstance(args[0], Sequence) and (not isinstance(args[0], str)):
        args = tuple(args[0])
    _logScreenshot(kwargs.get('logScreenshot'), 'hotkey', ','.join(args), folder='.')
    for c in args:
        if len(c) > 1:
            c = c.lower()
        platformModule._keyDown(c)
        time.sleep(interval)
    for c in reversed(args):
        if len(c) > 1:
            c = c.lower()
        platformModule._keyUp(c)
        time.sleep(interval)