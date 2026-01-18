from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def read_cursor_position(self, keys, more_available: bool):
    """
        Interpret cursor position information being sent by the
        user's terminal.  Returned as ('cursor position', x, y)
        where (x, y) == (0, 0) is the top left of the screen.
        """
    if not keys:
        if more_available:
            raise MoreInputRequired()
        return None
    if keys[0] != ord('['):
        return None
    y = 0
    i = 1
    for k in keys[i:]:
        i += 1
        if k == ord(';'):
            if not y:
                return None
            break
        if k < ord('0') or k > ord('9'):
            return None
        if not y and k == ord('0'):
            return None
        y = y * 10 + k - ord('0')
    if not keys[i:]:
        if more_available:
            raise MoreInputRequired()
        return None
    x = 0
    for k in keys[i:]:
        i += 1
        if k == ord('R'):
            if not x:
                return None
            return (('cursor position', x - 1, y - 1), keys[i:])
        if k < ord('0') or k > ord('9'):
            return None
        if not x and k == ord('0'):
            return None
        x = x * 10 + k - ord('0')
    if not keys[i:] and more_available:
        raise MoreInputRequired()
    return None