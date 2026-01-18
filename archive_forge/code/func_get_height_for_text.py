from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.search_state import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .lexers import Lexer, SimpleLexer
from .processors import Processor
from .screen import Char, Point
from .utils import token_list_width, split_lines, token_list_to_text
import six
import time
@staticmethod
def get_height_for_text(text, width):
    line_width = get_cwidth(text)
    try:
        quotient, remainder = divmod(line_width, width)
    except ZeroDivisionError:
        return 10 ** 10
    else:
        if remainder:
            quotient += 1
        return max(1, quotient)