from __future__ import unicode_literals
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import HasSelection, Condition, EmacsInsertMode, ViInsertMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.renderer import HeightIsUnknownError
from prompt_toolkit.utils import suspend_to_background_supported, is_windows
from .named_commands import get_by_name
from ..registry import Registry
def load_auto_suggestion_bindings():
    """
    Key bindings for accepting auto suggestion text.
    """
    registry = Registry()
    handle = registry.add_binding
    suggestion_available = Condition(lambda cli: cli.current_buffer.suggestion is not None and cli.current_buffer.document.is_cursor_at_the_end)

    @handle(Keys.ControlF, filter=suggestion_available)
    @handle(Keys.ControlE, filter=suggestion_available)
    @handle(Keys.Right, filter=suggestion_available)
    def _(event):
        """ Accept suggestion. """
        b = event.current_buffer
        suggestion = b.suggestion
        if suggestion:
            b.insert_text(suggestion.text)
    return registry