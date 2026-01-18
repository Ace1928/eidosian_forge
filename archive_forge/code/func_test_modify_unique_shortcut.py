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
def test_modify_unique_shortcut(ipython_with_prompt):
    original = find_bindings_by_command(accept_token)
    assert len(original) == 1
    ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_suggest.accept_token', 'new_keys': ['a', 'b', 'c']}]
    matched = find_bindings_by_command(accept_token)
    assert len(matched) == 1
    assert list(matched[0].keys) == ['a', 'b', 'c']
    assert list(matched[0].keys) != list(original[0].keys)
    assert matched[0].filter == original[0].filter
    ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_suggest.accept_token', 'new_filter': 'always'}]
    matched = find_bindings_by_command(accept_token)
    assert len(matched) == 1
    assert list(matched[0].keys) != ['a', 'b', 'c']
    assert list(matched[0].keys) == list(original[0].keys)
    assert matched[0].filter != original[0].filter