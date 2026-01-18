from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def start_of_paragraph(self, count=1, before=False):
    """
        Return the start of the current paragraph. (Relative cursor position.)
        """

    def match_func(text):
        return not text or text.isspace()
    line_index = self.find_previous_matching_line(match_func=match_func, count=count)
    if line_index:
        add = 0 if before else 1
        return min(0, self.get_cursor_up_position(count=-line_index) + add)
    else:
        return -self.cursor_position