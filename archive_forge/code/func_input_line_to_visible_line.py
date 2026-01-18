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
@property
def input_line_to_visible_line(self):
    """
        Return the dictionary mapping the line numbers of the input buffer to
        the lines of the screen. When a line spans several rows at the screen,
        the first row appears in the dictionary.
        """
    result = {}
    for k, v in self.visible_line_to_input_line.items():
        if v in result:
            result[v] = min(result[v], k)
        else:
            result[v] = k
    return result