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
def new_text_and_position(self):
    """
        Return (new_text, new_cursor_position) for this completion.
        """
    if self.complete_index is None:
        return (self.original_document.text, self.original_document.cursor_position)
    else:
        original_text_before_cursor = self.original_document.text_before_cursor
        original_text_after_cursor = self.original_document.text_after_cursor
        c = self.current_completions[self.complete_index]
        if c.start_position == 0:
            before = original_text_before_cursor
        else:
            before = original_text_before_cursor[:c.start_position]
        new_text = before + c.text + original_text_after_cursor
        new_cursor_position = len(before) + len(c.text)
        return (new_text, new_cursor_position)