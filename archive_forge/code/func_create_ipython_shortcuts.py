import os
import signal
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Any, Optional, List
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.vi_state import InputMode, ViState
from prompt_toolkit.filters import Condition
from IPython.core.getipython import get_ipython
from IPython.terminal.shortcuts import auto_match as match
from IPython.terminal.shortcuts import auto_suggest
from IPython.terminal.shortcuts.filters import filter_from_string
from IPython.utils.decorators import undoc
from prompt_toolkit.enums import DEFAULT_BUFFER
def create_ipython_shortcuts(shell, skip=None) -> KeyBindings:
    """Set up the prompt_toolkit keyboard shortcuts for IPython.

    Parameters
    ----------
    shell: InteractiveShell
        The current IPython shell Instance
    skip: List[Binding]
        Bindings to skip.

    Returns
    -------
    KeyBindings
        the keybinding instance for prompt toolkit.

    """
    kb = KeyBindings()
    skip = skip or []
    for binding in KEY_BINDINGS:
        skip_this_one = False
        for to_skip in skip:
            if to_skip.command == binding.command and to_skip.filter == binding.filter and (to_skip.keys == binding.keys):
                skip_this_one = True
                break
        if skip_this_one:
            continue
        add_binding(kb, binding)

    def get_input_mode(self):
        app = get_app()
        app.ttimeoutlen = shell.ttimeoutlen
        app.timeoutlen = shell.timeoutlen
        return self._input_mode

    def set_input_mode(self, mode):
        shape = {InputMode.NAVIGATION: 2, InputMode.REPLACE: 4}.get(mode, 6)
        cursor = '\x1b[{} q'.format(shape)
        sys.stdout.write(cursor)
        sys.stdout.flush()
        self._input_mode = mode
    if shell.editing_mode == 'vi' and shell.modal_cursor:
        ViState._input_mode = InputMode.INSERT
        ViState.input_mode = property(get_input_mode, set_input_mode)
    return kb