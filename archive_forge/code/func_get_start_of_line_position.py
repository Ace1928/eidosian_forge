from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_start_of_line_position(self, after_whitespace=False):
    """ Relative position for the start of this line. """
    if after_whitespace:
        current_line = self.current_line
        return len(current_line) - len(current_line.lstrip()) - self.cursor_position_col
    else:
        return -len(self.current_line_before_cursor)