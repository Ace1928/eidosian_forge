from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_end_of_line_position(self):
    """ Relative position for the end of this line. """
    return len(self.current_line_after_cursor)