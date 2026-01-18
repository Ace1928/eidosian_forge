from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_column_cursor_position(self, column):
    """
        Return the relative cursor position for this column at the current
        line. (It will stay between the boundaries of the line in case of a
        larger number.)
        """
    line_length = len(self.current_line)
    current_column = self.cursor_position_col
    column = max(0, min(line_length, column))
    return column - current_column