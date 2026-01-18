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
def create_operator_decorator(registry):
    """
    Create a decorator that can be used for registering Vi operators.
    """
    assert isinstance(registry, BaseRegistry)
    operator_given = ViWaitingForTextObjectMode()
    navigation_mode = ViNavigationMode()
    selection_mode = ViSelectionMode()

    def operator_decorator(*keys, **kw):
        """
        Register a Vi operator.

        Usage::

            @operator('d', filter=...)
            def handler(cli, text_object):
                # Do something with the text object here.
        """
        filter = kw.pop('filter', Always())
        eager = kw.pop('eager', False)
        assert not kw

        def decorator(operator_func):

            @registry.add_binding(*keys, filter=~operator_given & filter & navigation_mode, eager=eager)
            def _(event):
                """
                Handle operator in navigation mode.
                """
                event.cli.vi_state.operator_func = operator_func
                event.cli.vi_state.operator_arg = event.arg

            @registry.add_binding(*keys, filter=~operator_given & filter & selection_mode, eager=eager)
            def _(event):
                """
                Handle operator in selection mode.
                """
                buff = event.current_buffer
                selection_state = buff.selection_state
                if selection_state.type == SelectionType.LINES:
                    text_obj_type = TextObjectType.LINEWISE
                elif selection_state.type == SelectionType.BLOCK:
                    text_obj_type = TextObjectType.BLOCK
                else:
                    text_obj_type = TextObjectType.INCLUSIVE
                text_object = TextObject(selection_state.original_cursor_position - buff.cursor_position, type=text_obj_type)
                operator_func(event, text_object)
                buff.selection_state = None
            return operator_func
        return decorator
    return operator_decorator