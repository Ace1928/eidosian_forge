from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
def mouse_handler(cli, mouse_event):
    """ Wrapper around the mouse_handler of the `UIControl` that turns
            screen coordinates into line coordinates. """
    yx_to_rowcol = dict(((v, k) for k, v in rowcol_to_yx.items()))
    y = mouse_event.position.y
    x = mouse_event.position.x
    max_y = write_position.ypos + len(visible_line_to_row_col) - 1
    y = min(max_y, y)
    while x >= 0:
        try:
            row, col = yx_to_rowcol[y, x]
        except KeyError:
            x -= 1
        else:
            result = self.content.mouse_handler(cli, MouseEvent(position=Point(x=col, y=row), event_type=mouse_event.event_type))
            break
    else:
        result = self.content.mouse_handler(cli, MouseEvent(position=Point(x=0, y=0), event_type=mouse_event.event_type))
    if result == NotImplemented:
        return self._mouse_handler(cli, mouse_event)
    return result