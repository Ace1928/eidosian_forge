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
def create_prompt_layout(message='', lexer=None, is_password=False, reserve_space_for_menu=8, get_prompt_tokens=None, get_continuation_tokens=None, get_rprompt_tokens=None, get_bottom_toolbar_tokens=None, display_completions_in_columns=False, extra_input_processors=None, multiline=False, wrap_lines=True):
    """
    Create a :class:`.Container` instance for a prompt.

    :param message: Text to be used as prompt.
    :param lexer: :class:`~prompt_toolkit.layout.lexers.Lexer` to be used for
        the highlighting.
    :param is_password: `bool` or :class:`~prompt_toolkit.filters.CLIFilter`.
        When True, display input as '*'.
    :param reserve_space_for_menu: Space to be reserved for the menu. When >0,
        make sure that a minimal height is allocated in the terminal, in order
        to display the completion menu.
    :param get_prompt_tokens: An optional callable that returns the tokens to be
        shown in the menu. (To be used instead of a `message`.)
    :param get_continuation_tokens: An optional callable that takes a
        CommandLineInterface and width as input and returns a list of (Token,
        text) tuples to be used for the continuation.
    :param get_bottom_toolbar_tokens: An optional callable that returns the
        tokens for a toolbar at the bottom.
    :param display_completions_in_columns: `bool` or
        :class:`~prompt_toolkit.filters.CLIFilter`. Display the completions in
        multiple columns.
    :param multiline: `bool` or :class:`~prompt_toolkit.filters.CLIFilter`.
        When True, prefer a layout that is more adapted for multiline input.
        Text after newlines is automatically indented, and search/arg input is
        shown below the input, instead of replacing the prompt.
    :param wrap_lines: `bool` or :class:`~prompt_toolkit.filters.CLIFilter`.
        When True (the default), automatically wrap long lines instead of
        scrolling horizontally.
    """
    assert isinstance(message, text_type), 'Please provide a unicode string.'
    assert get_bottom_toolbar_tokens is None or callable(get_bottom_toolbar_tokens)
    assert get_prompt_tokens is None or callable(get_prompt_tokens)
    assert get_rprompt_tokens is None or callable(get_rprompt_tokens)
    assert not (message and get_prompt_tokens)
    display_completions_in_columns = to_cli_filter(display_completions_in_columns)
    multiline = to_cli_filter(multiline)
    if get_prompt_tokens is None:
        get_prompt_tokens = lambda _: [(Token.Prompt, message)]
    has_before_tokens, get_prompt_tokens_1, get_prompt_tokens_2 = _split_multiline_prompt(get_prompt_tokens)
    try:
        if pygments_Lexer and issubclass(lexer, pygments_Lexer):
            lexer = PygmentsLexer(lexer, sync_from_start=True)
    except TypeError:
        pass
    input_processors = [ConditionalProcessor(HighlightSearchProcessor(preview_search=True), HasFocus(SEARCH_BUFFER)), HighlightSelectionProcessor(), ConditionalProcessor(AppendAutoSuggestion(), HasFocus(DEFAULT_BUFFER) & ~IsDone()), ConditionalProcessor(PasswordProcessor(), is_password), DisplayMultipleCursors(DEFAULT_BUFFER)]
    if extra_input_processors:
        input_processors.extend(extra_input_processors)
    input_processors.append(ConditionalProcessor(DefaultPrompt(get_prompt_tokens_2), ~multiline))
    if get_bottom_toolbar_tokens:
        toolbars = [ConditionalContainer(Window(TokenListControl(get_bottom_toolbar_tokens, default_char=Char(' ', Token.Toolbar)), height=LayoutDimension.exact(1)), filter=~IsDone() & RendererHeightIsKnown())]
    else:
        toolbars = []

    def get_height(cli):
        if reserve_space_for_menu and (not cli.is_done):
            buff = cli.current_buffer
            if buff.complete_while_typing() or buff.complete_state is not None:
                return LayoutDimension(min=reserve_space_for_menu)
        return LayoutDimension()
    return HSplit([FloatContainer(HSplit([ConditionalContainer(Window(TokenListControl(get_prompt_tokens_1), dont_extend_height=True), Condition(has_before_tokens)), Window(BufferControl(input_processors=input_processors, lexer=lexer, preview_search=True), get_height=get_height, left_margins=[ConditionalMargin(PromptMargin(get_prompt_tokens_2, get_continuation_tokens), filter=multiline)], wrap_lines=wrap_lines)]), [Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=16, scroll_offset=1, extra_filter=HasFocus(DEFAULT_BUFFER) & ~display_completions_in_columns)), Float(xcursor=True, ycursor=True, content=MultiColumnCompletionsMenu(extra_filter=HasFocus(DEFAULT_BUFFER) & display_completions_in_columns, show_meta=True)), Float(right=0, top=0, hide_when_covering_content=True, content=_RPrompt(get_rprompt_tokens))]), ValidationToolbar(), SystemToolbar(), ConditionalContainer(ArgToolbar(), multiline), ConditionalContainer(SearchToolbar(), multiline)] + toolbars)