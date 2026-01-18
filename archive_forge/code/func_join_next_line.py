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
def join_next_line(self, separator=' '):
    """
        Join the next line to the current one by deleting the line ending after
        the current line.
        """
    if not self.document.on_last_line:
        self.cursor_position += self.document.get_end_of_line_position()
        self.delete()
        self.text = self.document.text_before_cursor + separator + self.document.text_after_cursor.lstrip(' ')