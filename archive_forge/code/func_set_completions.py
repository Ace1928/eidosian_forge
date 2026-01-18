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
def set_completions(self, completions, go_to_first=True, go_to_last=False):
    """
        Start completions. (Generate list of completions and initialize.)
        """
    assert not (go_to_first and go_to_last)
    if completions is None:
        if self.completer:
            completions = list(self.completer.get_completions(self.document, CompleteEvent(completion_requested=True)))
        else:
            completions = []
    if completions:
        self.complete_state = CompletionState(original_document=self.document, current_completions=completions)
        if go_to_first:
            self.go_to_completion(0)
        elif go_to_last:
            self.go_to_completion(len(completions) - 1)
        else:
            self.go_to_completion(None)
    else:
        self.complete_state = None