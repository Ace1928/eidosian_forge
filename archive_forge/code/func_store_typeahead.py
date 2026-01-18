from __future__ import annotations
from collections import defaultdict
from ..key_binding import KeyPress
from .base import Input
def store_typeahead(input_obj: Input, key_presses: list[KeyPress]) -> None:
    """
    Insert typeahead key presses for the given input.
    """
    global _buffer
    key = input_obj.typeahead_hash()
    _buffer[key].extend(key_presses)