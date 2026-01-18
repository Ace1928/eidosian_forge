import os
import sys
import inspect
from warnings import warn
from typing import Union as UnionType, Optional
from IPython.core.async_helpers import get_asyncio_loop
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.utils.py3compat import input
from IPython.utils.terminal import toggle_set_term_title, set_term_title, restore_term_title
from IPython.utils.process import abbrev_cwd
from traitlets import (
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import HasFocus, Condition, IsDone
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import History
from prompt_toolkit.layout.processors import ConditionalProcessor, HighlightMatchingBracketProcessor
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, CompleteStyle, print_formatted_text
from prompt_toolkit.styles import DynamicStyle, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_cls, style_from_pygments_dict
from prompt_toolkit import __version__ as ptk_version
from pygments.styles import get_style_by_name
from pygments.style import Style
from pygments.token import Token
from .debugger import TerminalPdb, Pdb
from .magics import TerminalMagics
from .pt_inputhooks import get_inputhook_name_and_func
from .prompts import Prompts, ClassicPrompts, RichPromptDisplayHook
from .ptutils import IPythonPTCompleter, IPythonPTLexer
from .shortcuts import (
from .shortcuts.filters import KEYBINDING_FILTERS, filter_from_string
from .shortcuts.auto_suggest import (
def init_prompt_toolkit_cli(self):
    if self.simple_prompt:

        def prompt():
            prompt_text = ''.join((x[1] for x in self.prompts.in_prompt_tokens()))
            lines = [input(prompt_text)]
            prompt_continuation = ''.join((x[1] for x in self.prompts.continuation_prompt_tokens()))
            while self.check_complete('\n'.join(lines))[0] == 'incomplete':
                lines.append(input(prompt_continuation))
            return '\n'.join(lines)
        self.prompt_for_code = prompt
        return
    key_bindings = self._merge_shortcuts(user_shortcuts=self.shortcuts)
    history = PtkHistoryAdapter(self)
    self._style = self._make_style_from_name_or_cls(self.highlighting_style)
    self.style = DynamicStyle(lambda: self._style)
    editing_mode = getattr(EditingMode, self.editing_mode.upper())
    self._use_asyncio_inputhook = False
    self.pt_app = PromptSession(auto_suggest=self.auto_suggest, editing_mode=editing_mode, key_bindings=key_bindings, history=history, completer=IPythonPTCompleter(shell=self), enable_history_search=self.enable_history_search, style=self.style, include_default_pygments_style=False, mouse_support=self.mouse_support, enable_open_in_editor=self.extra_open_editor_shortcuts, color_depth=self.color_depth, tempfile_suffix='.py', **self._extra_prompt_options())
    if isinstance(self.auto_suggest, NavigableAutoSuggestFromHistory):
        self.auto_suggest.connect(self.pt_app)