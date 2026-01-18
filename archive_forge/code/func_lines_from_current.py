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
def lines_from_current(self):
    """
        Array of the lines starting from the current line, until the last line.
        """
    return self.lines[self.cursor_position_row:]