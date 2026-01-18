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
def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
    """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
    if self.complete_state:
        self.complete_previous(count=count)
    elif self.document.cursor_position_row > 0:
        self.cursor_up(count=count)
    elif not self.selection_state:
        self.history_backward(count=count)
        if go_to_start_of_line_if_history_changes:
            self.cursor_position += self.document.get_start_of_line_position()