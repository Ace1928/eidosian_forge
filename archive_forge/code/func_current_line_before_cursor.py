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
def current_line_before_cursor(self):
    """ Text from the start of the line until the cursor. """
    _, _, text = self.text_before_cursor.rpartition('\n')
    return text