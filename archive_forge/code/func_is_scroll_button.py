from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
def is_scroll_button(row):
    """ True if we should display a button on this row. """
    current_row_middle = int((row + 0.5) * items_per_row)
    return current_row_middle in window_render_info.displayed_lines