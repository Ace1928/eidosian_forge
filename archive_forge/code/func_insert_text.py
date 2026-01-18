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
def insert_text(self, data, overwrite=False, move_cursor=True, fire_event=True):
    """
        Insert characters at cursor position.

        :param fire_event: Fire `on_text_insert` event. This is mainly used to
            trigger autocompletion while typing.
        """
    otext = self.text
    ocpos = self.cursor_position
    if overwrite:
        overwritten_text = otext[ocpos:ocpos + len(data)]
        if '\n' in overwritten_text:
            overwritten_text = overwritten_text[:overwritten_text.find('\n')]
        self.text = otext[:ocpos] + data + otext[ocpos + len(overwritten_text):]
    else:
        self.text = otext[:ocpos] + data + otext[ocpos:]
    if move_cursor:
        self.cursor_position += len(data)
    if fire_event:
        self.on_text_insert.fire()