from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_start_of_document_position(self):
    """ Relative position for the start of the document. """
    return -self.cursor_position