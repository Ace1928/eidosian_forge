from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def emacs_insert_mode() -> bool:
    app = get_app()
    if app.editing_mode != EditingMode.EMACS or app.current_buffer.selection_state or app.current_buffer.read_only():
        return False
    return True