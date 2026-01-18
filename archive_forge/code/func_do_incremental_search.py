from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING
from .application.current import get_app
from .filters import FilterOrBool, is_searching, to_filter
from .key_binding.vi_state import InputMode
def do_incremental_search(direction: SearchDirection, count: int=1) -> None:
    """
    Apply search, but keep search buffer focused.
    """
    assert is_searching()
    layout = get_app().layout
    from prompt_toolkit.layout.controls import BufferControl
    search_control = layout.current_control
    if not isinstance(search_control, BufferControl):
        return
    prev_control = layout.search_target_buffer_control
    if prev_control is None:
        return
    search_state = prev_control.search_state
    direction_changed = search_state.direction != direction
    search_state.text = search_control.buffer.text
    search_state.direction = direction
    if not direction_changed:
        prev_control.buffer.apply_search(search_state, include_current_position=False, count=count)