from __future__ import annotations
from collections import defaultdict
from ..key_binding import KeyPress
from .base import Input
def clear_typeahead(input_obj: Input) -> None:
    """
    Clear typeahead buffer.
    """
    global _buffer
    key = input_obj.typeahead_hash()
    _buffer[key] = []