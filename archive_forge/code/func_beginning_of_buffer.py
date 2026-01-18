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
@register('beginning-of-buffer')
def beginning_of_buffer(event: E) -> None:
    """
    Move to the start of the buffer.
    """
    buff = event.current_buffer
    buff.cursor_position = 0