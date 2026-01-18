from __future__ import unicode_literals
from prompt_toolkit.buffer import SelectionType, indent, unindent
from prompt_toolkit.keys import Keys
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Condition, EmacsMode, HasSelection, EmacsInsertMode, HasFocus, HasArg
from prompt_toolkit.completion import CompleteEvent
from .scroll import scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry
def load_extra_emacs_page_navigation_bindings():
    """
    Key bindings, for scrolling up and down through pages.
    This are separate bindings, because GNU readline doesn't have them.
    """
    registry = ConditionalRegistry(Registry(), EmacsMode())
    handle = registry.add_binding
    handle(Keys.ControlV)(scroll_page_down)
    handle(Keys.PageDown)(scroll_page_down)
    handle(Keys.Escape, 'v')(scroll_page_up)
    handle(Keys.PageUp)(scroll_page_up)
    return registry