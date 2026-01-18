from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_word_before_cursor(self, WORD=False):
    """
        Give the word before the cursor.
        If we have whitespace before the cursor this returns an empty string.
        """
    if self.text_before_cursor[-1:].isspace():
        return ''
    else:
        return self.text_before_cursor[self.find_start_of_previous_word(WORD=WORD):]