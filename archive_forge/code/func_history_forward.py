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
def history_forward(self, count=1):
    """
        Move forwards through the history.

        :param count: Amount of items to move forward.
        """
    self._set_history_search()
    found_something = False
    for i in range(self.working_index + 1, len(self._working_lines)):
        if self._history_matches(i):
            self.working_index = i
            count -= 1
            found_something = True
        if count == 0:
            break
    if found_something:
        self.cursor_position = 0
        self.cursor_position += self.document.get_end_of_line_position()