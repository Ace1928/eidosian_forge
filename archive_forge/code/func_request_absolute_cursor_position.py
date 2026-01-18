from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
def request_absolute_cursor_position(self):
    """
        Get current cursor position.
        For vt100: Do CPR request. (answer will arrive later.)
        For win32: Do API call. (Answer comes immediately.)
        """
    assert self._cursor_pos.y == 0
    if is_windows():
        self._min_available_height = self.output.get_rows_below_cursor_position()
    elif self.use_alternate_screen:
        self._min_available_height = self.output.get_size().rows
    else:
        self.waiting_for_cpr = True
        self.output.ask_for_cpr()