from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
@property
def cursor_position_col(self):
    """
        Current column. (0-based.)
        """
    _, line_start_index = self._find_line_start_index(self.cursor_position)
    return self.cursor_position - line_start_index