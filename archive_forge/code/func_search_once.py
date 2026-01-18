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
def search_once(working_index, document):
    """
            Do search one time.
            Return (working_index, document) or `None`
            """
    if direction == IncrementalSearchDirection.FORWARD:
        new_index = document.find(text, include_current_position=include_current_position, ignore_case=ignore_case)
        if new_index is not None:
            return (working_index, Document(document.text, document.cursor_position + new_index))
        else:
            for i in range(working_index + 1, len(self._working_lines) + 1):
                i %= len(self._working_lines)
                document = Document(self._working_lines[i], 0)
                new_index = document.find(text, include_current_position=True, ignore_case=ignore_case)
                if new_index is not None:
                    return (i, Document(document.text, new_index))
    else:
        new_index = document.find_backwards(text, ignore_case=ignore_case)
        if new_index is not None:
            return (working_index, Document(document.text, document.cursor_position + new_index))
        else:
            for i in range(working_index - 1, -2, -1):
                i %= len(self._working_lines)
                document = Document(self._working_lines[i], len(self._working_lines[i]))
                new_index = document.find_backwards(text, ignore_case=ignore_case)
                if new_index is not None:
                    return (i, Document(document.text, len(document.text) + new_index))