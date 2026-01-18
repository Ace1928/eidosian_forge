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
def current_char(self):
    """ Return character under cursor or an empty string. """
    return self._get_char_relative_to_cursor(0) or ''