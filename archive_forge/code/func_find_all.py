from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_all(self, sub, ignore_case=False):
    """
        Find all occurances of the substring. Return a list of absolute
        positions in the document.
        """
    flags = re.IGNORECASE if ignore_case else 0
    return [a.start() for a in re.finditer(re.escape(sub), self.text, flags)]