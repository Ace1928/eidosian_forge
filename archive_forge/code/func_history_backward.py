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
def history_backward(self, count=1):
    """
        Move backwards through history.
        """
    self._set_history_search()
    found_something = False
    for i in range(self.working_index - 1, -1, -1):
        if self._history_matches(i):
            self.working_index = i
            count -= 1
            found_something = True
        if count == 0:
            break
    if found_something:
        self.cursor_position = len(self.text)