from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def emacs_mode() -> bool:
    """When the Emacs bindings are active."""
    return get_app().editing_mode == EditingMode.EMACS