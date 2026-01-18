from __future__ import unicode_literals
from .buffer import Buffer, AcceptAction
from .document import Document
from .enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from .filters import IsDone, HasFocus, RendererHeightIsKnown, to_simple_filter, to_cli_filter, Condition
from .history import InMemoryHistory
from .interface import CommandLineInterface, Application, AbortAction
from .key_binding.defaults import load_key_bindings_for_prompt
from .key_binding.registry import Registry
from .keys import Keys
from .layout import Window, HSplit, FloatContainer, Float
from .layout.containers import ConditionalContainer
from .layout.controls import BufferControl, TokenListControl
from .layout.dimension import LayoutDimension
from .layout.lexers import PygmentsLexer
from .layout.margins import PromptMargin, ConditionalMargin
from .layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from .layout.processors import PasswordProcessor, ConditionalProcessor, AppendAutoSuggestion, HighlightSearchProcessor, HighlightSelectionProcessor, DisplayMultipleCursors
from .layout.prompt import DefaultPrompt
from .layout.screen import Char
from .layout.toolbars import ValidationToolbar, SystemToolbar, ArgToolbar, SearchToolbar
from .layout.utils import explode_tokens
from .renderer import print_tokens as renderer_print_tokens
from .styles import DEFAULT_STYLE, Style, style_from_dict
from .token import Token
from .utils import is_conemu_ansi, is_windows, DummyContext
from six import text_type, exec_, PY2
import os
import sys
import textwrap
import threading
import time
def create_output(stdout=None, true_color=False, ansi_colors_only=None):
    """
    Return an :class:`~prompt_toolkit.output.Output` instance for the command
    line.

    :param true_color: When True, use 24bit colors instead of 256 colors.
        (`bool` or :class:`~prompt_toolkit.filters.SimpleFilter`.)
    :param ansi_colors_only: When True, restrict to 16 ANSI colors only.
        (`bool` or :class:`~prompt_toolkit.filters.SimpleFilter`.)
    """
    stdout = stdout or sys.__stdout__
    true_color = to_simple_filter(true_color)
    if is_windows():
        if is_conemu_ansi():
            return ConEmuOutput(stdout)
        else:
            return Win32Output(stdout)
    else:
        term = os.environ.get('TERM', '')
        if PY2:
            term = term.decode('utf-8')
        return Vt100_Output.from_pty(stdout, true_color=true_color, ansi_colors_only=ansi_colors_only, term=term)