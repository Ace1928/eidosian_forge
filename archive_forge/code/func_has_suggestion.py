from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def has_suggestion() -> bool:
    """
    Enable when the current buffer has a suggestion.
    """
    buffer = get_app().current_buffer
    return buffer.suggestion is not None and buffer.suggestion.text != ''