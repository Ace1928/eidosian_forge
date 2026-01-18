from __future__ import annotations
from typing import Callable, TypeVar, Union, cast
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.key_bindings import Binding, key_binding
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.selection import PasteMode
from .completion import display_completions_like_readline, generate_completions
@register('call-last-kbd-macro')
@key_binding(record_in_macro=False)
def call_last_kbd_macro(event: E) -> None:
    """
    Re-execute the last keyboard macro defined, by making the characters in the
    macro appear as if typed at the keyboard.

    Notice that we pass `record_in_macro=False`. This ensures that the 'c-x e'
    key sequence doesn't appear in the recording itself. This function inserts
    the body of the called macro back into the KeyProcessor, so these keys will
    be added later on to the macro of their handlers have `record_in_macro=True`.
    """
    macro = event.app.emacs_state.macro
    if macro:
        event.app.key_processor.feed_multiple(macro, first=True)