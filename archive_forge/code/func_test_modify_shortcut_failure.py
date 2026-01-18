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
def test_modify_shortcut_failure(ipython_with_prompt):
    with pytest.raises(ValueError, match='No shortcuts matching'):
        ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_match.skip_over', 'match_keys': ['x'], 'new_keys': ['y']}]