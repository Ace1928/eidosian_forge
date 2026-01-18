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
def create_text_object_decorator(registry):
    """
    Create a decorator that can be used to register Vi text object implementations.
    """
    assert isinstance(registry, BaseRegistry)
    operator_given = ViWaitingForTextObjectMode()
    navigation_mode = ViNavigationMode()
    selection_mode = ViSelectionMode()

    def text_object_decorator(*keys, **kw):
        """
        Register a text object function.

        Usage::

            @text_object('w', filter=..., no_move_handler=False)
            def handler(event):
                # Return a text object for this key.
                return TextObject(...)

        :param no_move_handler: Disable the move handler in navigation mode.
            (It's still active in selection mode.)
        """
        filter = kw.pop('filter', Always())
        no_move_handler = kw.pop('no_move_handler', False)
        no_selection_handler = kw.pop('no_selection_handler', False)
        eager = kw.pop('eager', False)
        assert not kw

        def decorator(text_object_func):
            assert callable(text_object_func)

            @registry.add_binding(*keys, filter=operator_given & filter, eager=eager)
            def _(event):
                vi_state = event.cli.vi_state
                event._arg = (vi_state.operator_arg or 1) * (event.arg or 1)
                text_obj = text_object_func(event)
                if text_obj is not None:
                    assert isinstance(text_obj, TextObject)
                    vi_state.operator_func(event, text_obj)
                event.cli.vi_state.operator_func = None
                event.cli.vi_state.operator_arg = None
            if not no_move_handler:

                @registry.add_binding(*keys, filter=~operator_given & filter & navigation_mode, eager=eager)
                def _(event):
                    """ Move handler for navigation mode. """
                    text_object = text_object_func(event)
                    event.current_buffer.cursor_position += text_object.start
            if not no_selection_handler:

                @registry.add_binding(*keys, filter=~operator_given & filter & selection_mode, eager=eager)
                def _(event):
                    """ Move handler for selection mode. """
                    text_object = text_object_func(event)
                    buff = event.current_buffer
                    if text_object.end:
                        start, end = text_object.operator_range(buff.document)
                        start += buff.cursor_position
                        end += buff.cursor_position
                        buff.selection_state.original_cursor_position = start
                        buff.cursor_position = end
                        if text_object.type == TextObjectType.LINEWISE:
                            buff.selection_state.type = SelectionType.LINES
                        else:
                            buff.selection_state.type = SelectionType.CHARACTERS
                    else:
                        event.current_buffer.cursor_position += text_object.start
            return text_object_func
        return decorator
    return text_object_decorator