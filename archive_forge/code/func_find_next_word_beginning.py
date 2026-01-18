from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_next_word_beginning(self, count=1, WORD=False):
    """
        Return an index relative to the cursor position pointing to the start
        of the next word. Return `None` if nothing was found.
        """
    if count < 0:
        return self.find_previous_word_beginning(count=-count, WORD=WORD)
    regex = _FIND_BIG_WORD_RE if WORD else _FIND_WORD_RE
    iterator = regex.finditer(self.text_after_cursor)
    try:
        for i, match in enumerate(iterator):
            if i == 0 and match.start(1) == 0:
                count += 1
            if i + 1 == count:
                return match.start(1)
    except StopIteration:
        pass