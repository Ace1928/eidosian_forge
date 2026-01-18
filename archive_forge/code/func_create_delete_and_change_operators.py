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
def create_delete_and_change_operators(delete_only, with_register=False):
    """
        Delete and change operators.

        :param delete_only: Create an operator that deletes, but doesn't go to insert mode.
        :param with_register: Copy the deleted text to this named register instead of the clipboard.
        """
    if with_register:
        handler_keys = ('"', Keys.Any, 'cd'[delete_only])
    else:
        handler_keys = 'cd'[delete_only]

    @operator(*handler_keys, filter=~IsReadOnly())
    def delete_or_change_operator(event, text_object):
        clipboard_data = None
        buff = event.current_buffer
        if text_object:
            new_document, clipboard_data = text_object.cut(buff)
            buff.document = new_document
        if clipboard_data and clipboard_data.text:
            if with_register:
                reg_name = event.key_sequence[1].data
                if reg_name in vi_register_names:
                    event.cli.vi_state.named_registers[reg_name] = clipboard_data
            else:
                event.cli.clipboard.set_data(clipboard_data)
        if not delete_only:
            event.cli.vi_state.input_mode = InputMode.INSERT