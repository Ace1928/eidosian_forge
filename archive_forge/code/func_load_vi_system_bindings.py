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
def load_vi_system_bindings():
    registry = ConditionalRegistry(Registry(), ViMode())
    handle = registry.add_binding
    has_focus = filters.HasFocus(SYSTEM_BUFFER)
    navigation_mode = ViNavigationMode()

    @handle('!', filter=~has_focus & navigation_mode)
    def _(event):
        """
        '!' opens the system prompt.
        """
        event.cli.push_focus(SYSTEM_BUFFER)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle(Keys.Escape, filter=has_focus)
    @handle(Keys.ControlC, filter=has_focus)
    def _(event):
        """
        Cancel system prompt.
        """
        event.cli.vi_state.input_mode = InputMode.NAVIGATION
        event.cli.buffers[SYSTEM_BUFFER].reset()
        event.cli.pop_focus()

    @handle(Keys.ControlJ, filter=has_focus)
    def _(event):
        """
        Run system command.
        """
        event.cli.vi_state.input_mode = InputMode.NAVIGATION
        system_buffer = event.cli.buffers[SYSTEM_BUFFER]
        event.cli.run_system_command(system_buffer.text)
        system_buffer.reset(append_to_history=True)
        event.cli.pop_focus()
    return registry