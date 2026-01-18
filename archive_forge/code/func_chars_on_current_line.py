import sys, os, unicodedata
import py
from py.builtin import text, bytes
@property
def chars_on_current_line(self):
    """Return the number of characters written so far in the current line.

        Please note that this count does not produce correct results after a reline() call,
        see #164.

        .. versionadded:: 1.5.0

        :rtype: int
        """
    return self._chars_on_current_line