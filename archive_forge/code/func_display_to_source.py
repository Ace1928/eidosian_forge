from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.enums import SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter, ViInsertMultipleMode
from prompt_toolkit.layout.utils import token_list_to_text
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from .utils import token_list_len, explode_tokens
import re
def display_to_source(display_pos):
    """ Maps display cursor position to the original one. """
    position_mappings_reversed = dict(((v, k) for k, v in position_mappings.items()))
    while display_pos >= 0:
        try:
            return position_mappings_reversed[display_pos]
        except KeyError:
            display_pos -= 1
    return 0