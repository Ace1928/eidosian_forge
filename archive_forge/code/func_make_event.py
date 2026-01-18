import pytest
from IPython.terminal.shortcuts.auto_suggest import (
from IPython.terminal.shortcuts.auto_match import skip_over
from IPython.terminal.shortcuts import create_ipython_shortcuts, reset_search_buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER
from unittest.mock import patch, Mock
def make_event(text, cursor, suggestion):
    event = Mock()
    event.current_buffer = Mock()
    event.current_buffer.suggestion = Mock()
    event.current_buffer.text = text
    event.current_buffer.cursor_position = cursor
    event.current_buffer.suggestion.text = suggestion
    event.current_buffer.document = Document(text=text, cursor_position=cursor)
    return event