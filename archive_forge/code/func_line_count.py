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
def line_count(self):
    """ Return the number of lines in this document. If the document ends
        with a trailing \\n, that counts as the beginning of a new line. """
    return len(self.lines)