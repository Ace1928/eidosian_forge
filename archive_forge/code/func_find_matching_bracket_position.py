from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_matching_bracket_position(self, start_pos=None, end_pos=None):
    """
        Return relative cursor position of matching [, (, { or < bracket.

        When `start_pos` or `end_pos` are given. Don't look past the positions.
        """
    for A, B in ('()', '[]', '{}', '<>'):
        if self.current_char == A:
            return self.find_enclosing_bracket_right(A, B, end_pos=end_pos) or 0
        elif self.current_char == B:
            return self.find_enclosing_bracket_left(A, B, start_pos=start_pos) or 0
    return 0