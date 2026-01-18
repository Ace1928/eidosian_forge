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
def on_first_line(self):
    """
        True when we are at the first line.
        """
    return self.cursor_position_row == 0