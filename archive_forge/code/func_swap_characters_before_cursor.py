from __future__ import unicode_literals
from .auto_suggest import AutoSuggest
from .clipboard import ClipboardData
from .completion import Completer, Completion, CompleteEvent
from .document import Document
from .enums import IncrementalSearchDirection
from .filters import to_simple_filter
from .history import History, InMemoryHistory
from .search_state import SearchState
from .selection import SelectionType, SelectionState, PasteMode
from .utils import Event
from .cache import FastDictCache
from .validation import ValidationError
from six.moves import range
import os
import re
import six
import subprocess
import tempfile
def swap_characters_before_cursor(self):
    """
        Swap the last two characters before the cursor.
        """
    pos = self.cursor_position
    if pos >= 2:
        a = self.text[pos - 2]
        b = self.text[pos - 1]
        self.text = self.text[:pos - 2] + b + a + self.text[pos:]