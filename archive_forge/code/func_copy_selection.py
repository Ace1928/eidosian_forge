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
def copy_selection(self, _cut=False):
    """
        Copy selected text and return :class:`.ClipboardData` instance.
        """
    new_document, clipboard_data = self.document.cut_selection()
    if _cut:
        self.document = new_document
    self.selection_state = None
    return clipboard_data