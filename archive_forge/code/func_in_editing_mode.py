from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@memoized()
def in_editing_mode(editing_mode: EditingMode) -> Condition:
    """
    Check whether a given editing mode is active. (Vi or Emacs.)
    """

    @Condition
    def in_editing_mode_filter() -> bool:
        return get_app().editing_mode == editing_mode
    return in_editing_mode_filter