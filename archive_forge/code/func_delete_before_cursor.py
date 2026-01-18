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
def delete_before_cursor(self, count=1):
    """
        Delete specified number of characters before cursor and return the
        deleted text.
        """
    assert count >= 0
    deleted = ''
    if self.cursor_position > 0:
        deleted = self.text[self.cursor_position - count:self.cursor_position]
        new_text = self.text[:self.cursor_position - count] + self.text[self.cursor_position:]
        new_cursor_position = self.cursor_position - len(deleted)
        self.document = Document(new_text, new_cursor_position)
    return deleted