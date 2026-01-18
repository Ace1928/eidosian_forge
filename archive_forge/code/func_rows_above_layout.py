from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
@property
def rows_above_layout(self):
    """
        Return the number of rows visible in the terminal above the layout.
        """
    if self._in_alternate_screen:
        return 0
    elif self._min_available_height > 0:
        total_rows = self.output.get_size().rows
        last_screen_height = self._last_screen.height if self._last_screen else 0
        return total_rows - max(self._min_available_height, last_screen_height)
    else:
        raise HeightIsUnknownError('Rows above layout is unknown.')