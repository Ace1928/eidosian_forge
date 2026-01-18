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
def load_vi_search_bindings(get_search_state=None, search_buffer_name=SEARCH_BUFFER):
    assert get_search_state is None or callable(get_search_state)
    if not get_search_state:

        def get_search_state(cli):
            return cli.search_state
    registry = ConditionalRegistry(Registry(), ViMode())
    handle = registry.add_binding
    has_focus = filters.HasFocus(search_buffer_name)
    navigation_mode = ViNavigationMode()
    selection_mode = ViSelectionMode()
    reverse_vi_search_direction = Condition(lambda cli: cli.application.reverse_vi_search_direction(cli))

    @handle('/', filter=(navigation_mode | selection_mode) & ~reverse_vi_search_direction)
    @handle('?', filter=(navigation_mode | selection_mode) & reverse_vi_search_direction)
    @handle(Keys.ControlS, filter=~has_focus)
    def _(event):
        """
        Vi-style forward search.
        """
        get_search_state(event.cli).direction = IncrementalSearchDirection.FORWARD
        event.cli.vi_state.input_mode = InputMode.INSERT
        event.cli.push_focus(search_buffer_name)

    @handle('?', filter=(navigation_mode | selection_mode) & ~reverse_vi_search_direction)
    @handle('/', filter=(navigation_mode | selection_mode) & reverse_vi_search_direction)
    @handle(Keys.ControlR, filter=~has_focus)
    def _(event):
        """
        Vi-style backward search.
        """
        get_search_state(event.cli).direction = IncrementalSearchDirection.BACKWARD
        event.cli.push_focus(search_buffer_name)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle(Keys.ControlJ, filter=has_focus)
    def _(event):
        """
        Apply the search. (At the / or ? prompt.)
        """
        input_buffer = event.cli.buffers.previous(event.cli)
        search_buffer = event.cli.buffers[search_buffer_name]
        if search_buffer.text:
            get_search_state(event.cli).text = search_buffer.text
        input_buffer.apply_search(get_search_state(event.cli))
        search_buffer.append_to_history()
        search_buffer.reset()
        event.cli.vi_state.input_mode = InputMode.NAVIGATION
        event.cli.pop_focus()

    def incremental_search(cli, direction, count=1):
        """ Apply search, but keep search buffer focussed. """
        search_state = get_search_state(cli)
        direction_changed = search_state.direction != direction
        search_state.text = cli.buffers[search_buffer_name].text
        search_state.direction = direction
        if not direction_changed:
            input_buffer = cli.buffers.previous(cli)
            input_buffer.apply_search(search_state, include_current_position=False, count=count)

    @handle(Keys.ControlR, filter=has_focus)
    def _(event):
        incremental_search(event.cli, IncrementalSearchDirection.BACKWARD, count=event.arg)

    @handle(Keys.ControlS, filter=has_focus)
    def _(event):
        incremental_search(event.cli, IncrementalSearchDirection.FORWARD, count=event.arg)

    def search_buffer_is_empty(cli):
        """ Returns True when the search buffer is empty. """
        return cli.buffers[search_buffer_name].text == ''

    @handle(Keys.Escape, filter=has_focus)
    @handle(Keys.ControlC, filter=has_focus)
    @handle(Keys.ControlH, filter=has_focus & Condition(search_buffer_is_empty))
    @handle(Keys.Backspace, filter=has_focus & Condition(search_buffer_is_empty))
    def _(event):
        """
        Cancel search.
        """
        event.cli.vi_state.input_mode = InputMode.NAVIGATION
        event.cli.pop_focus()
        event.cli.buffers[search_buffer_name].reset()
    return registry