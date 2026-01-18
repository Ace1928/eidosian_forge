from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_cursor_right_position(self, count=1):
    """
        Relative position for cursor_right.
        """
    if count < 0:
        return self.get_cursor_left_position(-count)
    return min(count, len(self.current_line_after_cursor))