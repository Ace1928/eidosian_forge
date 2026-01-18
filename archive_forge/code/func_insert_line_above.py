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
def insert_line_above(self, copy_margin=True):
    """
        Insert a new line above the current one.
        """
    if copy_margin:
        insert = self.document.leading_whitespace_in_current_line + '\n'
    else:
        insert = '\n'
    self.cursor_position += self.document.get_start_of_line_position()
    self.insert_text(insert)
    self.cursor_position -= 1