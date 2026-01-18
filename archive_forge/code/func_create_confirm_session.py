from __future__ import annotations
from asyncio import get_running_loop
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, Iterator, TypeVar, Union, cast
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import Clipboard, DynamicClipboard, InMemoryClipboard
from prompt_toolkit.completion import Completer, DynamicCompleter, ThreadedCompleter
from prompt_toolkit.cursor_shapes import (
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from prompt_toolkit.eventloop import InputHook
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding.bindings.auto_suggest import load_auto_suggest_bindings
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.bindings.open_in_editor import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, WindowAlign
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.layout.processors import (
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.lexers import DynamicLexer, Lexer
from prompt_toolkit.output import ColorDepth, DummyOutput, Output
from prompt_toolkit.styles import (
from prompt_toolkit.utils import (
from prompt_toolkit.validation import DynamicValidator, Validator
from prompt_toolkit.widgets.toolbars import (
def create_confirm_session(message: str, suffix: str=' (y/n) ') -> PromptSession[bool]:
    """
    Create a `PromptSession` object for the 'confirm' function.
    """
    bindings = KeyBindings()

    @bindings.add('y')
    @bindings.add('Y')
    def yes(event: E) -> None:
        session.default_buffer.text = 'y'
        event.app.exit(result=True)

    @bindings.add('n')
    @bindings.add('N')
    def no(event: E) -> None:
        session.default_buffer.text = 'n'
        event.app.exit(result=False)

    @bindings.add(Keys.Any)
    def _(event: E) -> None:
        """Disallow inserting other text."""
        pass
    complete_message = merge_formatted_text([message, suffix])
    session: PromptSession[bool] = PromptSession(complete_message, key_bindings=bindings)
    return session