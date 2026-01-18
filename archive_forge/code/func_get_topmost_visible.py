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
def get_topmost_visible():
    """
            Calculate the upper most line that can be visible, while the bottom
            is still visible. We should not allow scroll more than this if
            `allow_scroll_beyond_bottom` is false.
            """
    prev_lineno = ui_content.line_count - 1
    used_height = 0
    for lineno in range(ui_content.line_count - 1, -1, -1):
        used_height += ui_content.get_height_for_line(lineno, width)
        if used_height > height:
            return prev_lineno
        else:
            prev_lineno = lineno
    return prev_lineno