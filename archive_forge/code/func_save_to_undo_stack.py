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
def save_to_undo_stack(self, clear_redo_stack=True):
    """
        Safe current state (input text and cursor position), so that we can
        restore it by calling undo.
        """
    if self._undo_stack and self._undo_stack[-1][0] == self.text:
        self._undo_stack[-1] = (self._undo_stack[-1][0], self.cursor_position)
    else:
        self._undo_stack.append((self.text, self.cursor_position))
    if clear_redo_stack:
        self._redo_stack = []