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
def do_scroll(current_scroll, scroll_offset_start, scroll_offset_end, cursor_pos, window_size, content_size):
    """ Scrolling algorithm. Used for both horizontal and vertical scrolling. """
    scroll_offset_start = int(min(scroll_offset_start, window_size / 2, cursor_pos))
    scroll_offset_end = int(min(scroll_offset_end, window_size / 2, content_size - 1 - cursor_pos))
    if current_scroll < 0:
        current_scroll = 0
    if not self.allow_scroll_beyond_bottom(cli) and current_scroll > content_size - window_size:
        current_scroll = max(0, content_size - window_size)
    if current_scroll > cursor_pos - scroll_offset_start:
        current_scroll = max(0, cursor_pos - scroll_offset_start)
    if current_scroll < cursor_pos + 1 - window_size + scroll_offset_end:
        current_scroll = cursor_pos + 1 - window_size + scroll_offset_end
    return current_scroll