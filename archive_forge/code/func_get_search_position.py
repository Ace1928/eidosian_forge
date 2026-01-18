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
def get_search_position(self, search_state, include_current_position=True, count=1):
    """
        Get the cursor position for this search.
        (This operation won't change the `working_index`. It's won't go through
        the history. Vi text objects can't span multiple items.)
        """
    search_result = self._search(search_state, include_current_position=include_current_position, count=count)
    if search_result is None:
        return self.cursor_position
    else:
        working_index, cursor_position = search_result
        return cursor_position