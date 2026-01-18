from __future__ import unicode_literals
from prompt_toolkit.buffer import SelectionType, indent, unindent
from prompt_toolkit.keys import Keys
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Condition, EmacsMode, HasSelection, EmacsInsertMode, HasFocus, HasArg
from prompt_toolkit.completion import CompleteEvent
from .scroll import scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry
def handle_digit(c):
    """
        Handle input of arguments.
        The first number needs to be preceeded by escape.
        """

    @handle(c, filter=HasArg())
    @handle(Keys.Escape, c)
    def _(event):
        event.append_to_arg_count(c)