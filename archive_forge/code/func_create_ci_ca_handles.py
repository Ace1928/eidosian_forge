from __future__ import unicode_literals
from prompt_toolkit.buffer import ClipboardData, indent, unindent, reshape_text
from prompt_toolkit.document import Document
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Filter, Condition, HasArg, Always, IsReadOnly
from prompt_toolkit.filters.cli import ViNavigationMode, ViInsertMode, ViInsertMultipleMode, ViReplaceMode, ViSelectionMode, ViWaitingForTextObjectMode, ViDigraphMode, ViMode
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from prompt_toolkit.selection import SelectionType, SelectionState, PasteMode
from .scroll import scroll_forward, scroll_backward, scroll_half_page_up, scroll_half_page_down, scroll_one_line_up, scroll_one_line_down, scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry, BaseRegistry
import prompt_toolkit.filters as filters
from six.moves import range
import codecs
import six
import string
def create_ci_ca_handles(ci_start, ci_end, inner, key=None):
    """
        Delete/Change string between this start and stop character. But keep these characters.
        This implements all the ci", ci<, ci{, ci(, di", di<, ca", ca<, ... combinations.
        """

    def handler(event):
        if ci_start == ci_end:
            start = event.current_buffer.document.find_backwards(ci_start, in_current_line=False)
            end = event.current_buffer.document.find(ci_end, in_current_line=False)
        else:
            start = event.current_buffer.document.find_enclosing_bracket_left(ci_start, ci_end)
            end = event.current_buffer.document.find_enclosing_bracket_right(ci_start, ci_end)
        if start is not None and end is not None:
            offset = 0 if inner else 1
            return TextObject(start + 1 - offset, end + offset)
        else:
            return TextObject(0)
    if key is None:
        text_object('ai'[inner], ci_start, no_move_handler=True)(handler)
        text_object('ai'[inner], ci_end, no_move_handler=True)(handler)
    else:
        text_object('ai'[inner], key, no_move_handler=True)(handler)