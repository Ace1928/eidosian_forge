from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_enclosing_bracket_right(self, left_ch, right_ch, end_pos=None):
    """
        Find the right bracket enclosing current position. Return the relative
        position to the cursor position.

        When `end_pos` is given, don't look past the position.
        """
    if self.current_char == right_ch:
        return 0
    if end_pos is None:
        end_pos = len(self.text)
    else:
        end_pos = min(len(self.text), end_pos)
    stack = 1
    for i in range(self.cursor_position + 1, end_pos):
        c = self.text[i]
        if c == left_ch:
            stack += 1
        elif c == right_ch:
            stack -= 1
        if stack == 0:
            return i - self.cursor_position