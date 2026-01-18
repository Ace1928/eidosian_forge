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
def start_history_lines_completion(self):
    """
        Start a completion based on all the other lines in the document and the
        history.
        """
    found_completions = set()
    completions = []
    current_line = self.document.current_line_before_cursor.lstrip()
    for i, string in enumerate(self._working_lines):
        for j, l in enumerate(string.split('\n')):
            l = l.strip()
            if l and l.startswith(current_line):
                if l not in found_completions:
                    found_completions.add(l)
                    if i == self.working_index:
                        display_meta = 'Current, line %s' % (j + 1)
                    else:
                        display_meta = 'History %s, line %s' % (i + 1, j + 1)
                    completions.append(Completion(l, start_position=-len(current_line), display_meta=display_meta))
    self.set_completions(completions=completions[::-1])