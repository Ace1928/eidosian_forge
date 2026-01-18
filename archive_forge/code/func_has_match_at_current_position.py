from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def has_match_at_current_position(self, sub):
    """
        `True` when this substring is found at the cursor position.
        """
    return self.text.find(sub, self.cursor_position) == self.cursor_position