from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_cursor_down_position(self, count=1, preferred_column=None):
    """
        Return the relative cursor position (character index) where we would be if the
        user pressed the arrow-down button.

        :param preferred_column: When given, go to this column instead of
                                 staying at the current column.
        """
    assert count >= 1
    column = self.cursor_position_col if preferred_column is None else preferred_column
    return self.translate_row_col_to_index(self.cursor_position_row + count, column) - self.cursor_position