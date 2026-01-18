from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def control_is_searchable() -> bool:
    """When the current UIControl is searchable."""
    from prompt_toolkit.layout.controls import BufferControl
    control = get_app().layout.current_control
    return isinstance(control, BufferControl) and control.search_buffer_control is not None