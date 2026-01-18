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
@Condition
def ctrl_d_condition(cli):
    """ Ctrl-D binding is only active when the default buffer is selected
        and empty. """
    return cli.current_buffer_name == DEFAULT_BUFFER and (not cli.current_buffer.text)