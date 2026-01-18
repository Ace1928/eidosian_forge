from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_previous_matching_line(self, match_func, count=1):
    """
        Look upwards for empty lines.
        Return the line index, relative to the current line.
        """
    result = None
    for index, line in enumerate(self.lines[:self.cursor_position_row][::-1]):
        if match_func(line):
            result = -1 - index
            count -= 1
        if count == 0:
            break
    return result