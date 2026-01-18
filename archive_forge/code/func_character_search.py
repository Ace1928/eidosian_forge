from __future__ import unicode_literals
from prompt_toolkit.buffer import SelectionType, indent, unindent
from prompt_toolkit.keys import Keys
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Condition, EmacsMode, HasSelection, EmacsInsertMode, HasFocus, HasArg
from prompt_toolkit.completion import CompleteEvent
from .scroll import scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry
def character_search(buff, char, count):
    if count < 0:
        match = buff.document.find_backwards(char, in_current_line=True, count=-count)
    else:
        match = buff.document.find(char, in_current_line=True, count=count)
    if match is not None:
        buff.cursor_position += match