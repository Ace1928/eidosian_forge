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
def load_basic_system_bindings():
    """
    Basic system bindings (For both Emacs and Vi mode.)
    """
    registry = Registry()
    suspend_supported = Condition(lambda cli: suspend_to_background_supported())

    @registry.add_binding(Keys.ControlZ, filter=suspend_supported)
    def _(event):
        """
        Suspend process to background.
        """
        event.cli.suspend_to_background()
    return registry