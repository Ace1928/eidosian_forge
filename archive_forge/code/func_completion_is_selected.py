from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def completion_is_selected() -> bool:
    """
    True when the user selected a completion.
    """
    complete_state = get_app().current_buffer.complete_state
    return complete_state is not None and complete_state.current_completion is not None