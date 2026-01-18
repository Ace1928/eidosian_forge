from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_enclosing_bracket_left(self, left_ch, right_ch, start_pos=None):
    """
        Find the left bracket enclosing current position. Return the relative
        position to the cursor position.

        When `start_pos` is given, don't look past the position.
        """
    if self.current_char == left_ch:
        return 0
    if start_pos is None:
        start_pos = 0
    else:
        start_pos = max(0, start_pos)
    stack = 1
    for i in range(self.cursor_position - 1, start_pos - 1, -1):
        c = self.text[i]
        if c == right_ch:
            stack += 1
        elif c == left_ch:
            stack -= 1
        if stack == 0:
            return i - self.cursor_position