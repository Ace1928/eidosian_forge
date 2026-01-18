from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def emacs_selection_mode() -> bool:
    app = get_app()
    return bool(app.editing_mode == EditingMode.EMACS and app.current_buffer.selection_state)