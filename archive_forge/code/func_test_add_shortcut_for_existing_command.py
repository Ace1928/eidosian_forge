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
def test_add_shortcut_for_existing_command(ipython_with_prompt):
    matched = find_bindings_by_command(skip_over)
    assert len(matched) == 5
    with pytest.raises(ValueError, match='Cannot add a shortcut without keys'):
        ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_match.skip_over', 'new_keys': [], 'create': True}]
    ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_match.skip_over', 'new_keys': ['x'], 'create': True}]
    matched = find_bindings_by_command(skip_over)
    assert len(matched) == 6
    ipython_with_prompt.shortcuts = []
    matched = find_bindings_by_command(skip_over)
    assert len(matched) == 5