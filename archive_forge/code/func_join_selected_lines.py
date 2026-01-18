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
def join_selected_lines(self, separator=' '):
    """
        Join the selected lines.
        """
    assert self.selection_state
    from_, to = sorted([self.cursor_position, self.selection_state.original_cursor_position])
    before = self.text[:from_]
    lines = self.text[from_:to].splitlines()
    after = self.text[to:]
    lines = [l.lstrip(' ') + separator for l in lines]
    self.document = Document(text=before + ''.join(lines) + after, cursor_position=len(before + ''.join(lines[:-1])) - 1)