from __future__ import annotations
import functools
from asyncio import get_running_loop
from typing import Any, Callable, Sequence, TypeVar
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import AnyContainer, HSplit
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import (
def input_dialog(title: AnyFormattedText='', text: AnyFormattedText='', ok_text: str='OK', cancel_text: str='Cancel', completer: Completer | None=None, validator: Validator | None=None, password: FilterOrBool=False, style: BaseStyle | None=None, default: str='') -> Application[str]:
    """
    Display a text input box.
    Return the given text, or None when cancelled.
    """

    def accept(buf: Buffer) -> bool:
        get_app().layout.focus(ok_button)
        return True

    def ok_handler() -> None:
        get_app().exit(result=textfield.text)
    ok_button = Button(text=ok_text, handler=ok_handler)
    cancel_button = Button(text=cancel_text, handler=_return_none)
    textfield = TextArea(text=default, multiline=False, password=password, completer=completer, validator=validator, accept_handler=accept)
    dialog = Dialog(title=title, body=HSplit([Label(text=text, dont_extend_height=True), textfield, ValidationToolbar()], padding=D(preferred=1, max=1)), buttons=[ok_button, cancel_button], with_background=True)
    return _create_app(dialog, style)