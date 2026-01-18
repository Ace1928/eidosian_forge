from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
def move_cursor(new):
    """ Move cursor to this `new` point. Returns the given Point. """
    current_x, current_y = (current_pos.x, current_pos.y)
    if new.y > current_y:
        reset_attributes()
        write('\r\n' * (new.y - current_y))
        current_x = 0
        _output_cursor_forward(new.x)
        return new
    elif new.y < current_y:
        _output_cursor_up(current_y - new.y)
    if current_x >= width - 1:
        write('\r')
        _output_cursor_forward(new.x)
    elif new.x < current_x or current_x >= width - 1:
        _output_cursor_backward(current_x - new.x)
    elif new.x > current_x:
        _output_cursor_forward(new.x - current_x)
    return new