from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, unindent
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_bindings import Binding
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.selection import SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
def load_emacs_shift_selection_bindings() -> KeyBindingsBase:
    """
    Bindings to select text with shift + cursor movements
    """
    key_bindings = KeyBindings()
    handle = key_bindings.add

    def unshift_move(event: E) -> None:
        """
        Used for the shift selection mode. When called with
        a shift + movement key press event, moves the cursor
        as if shift is not pressed.
        """
        key = event.key_sequence[0].key
        if key == Keys.ShiftUp:
            event.current_buffer.auto_up(count=event.arg)
            return
        if key == Keys.ShiftDown:
            event.current_buffer.auto_down(count=event.arg)
            return
        key_to_command: dict[Keys | str, str] = {Keys.ShiftLeft: 'backward-char', Keys.ShiftRight: 'forward-char', Keys.ShiftHome: 'beginning-of-line', Keys.ShiftEnd: 'end-of-line', Keys.ControlShiftLeft: 'backward-word', Keys.ControlShiftRight: 'forward-word', Keys.ControlShiftHome: 'beginning-of-buffer', Keys.ControlShiftEnd: 'end-of-buffer'}
        try:
            binding = get_by_name(key_to_command[key])
        except KeyError:
            pass
        else:
            if isinstance(binding, Binding):
                binding.call(event)

    @handle('s-left', filter=~has_selection)
    @handle('s-right', filter=~has_selection)
    @handle('s-up', filter=~has_selection)
    @handle('s-down', filter=~has_selection)
    @handle('s-home', filter=~has_selection)
    @handle('s-end', filter=~has_selection)
    @handle('c-s-left', filter=~has_selection)
    @handle('c-s-right', filter=~has_selection)
    @handle('c-s-home', filter=~has_selection)
    @handle('c-s-end', filter=~has_selection)
    def _start_selection(event: E) -> None:
        """
        Start selection with shift + movement.
        """
        buff = event.current_buffer
        if buff.text:
            buff.start_selection(selection_type=SelectionType.CHARACTERS)
            if buff.selection_state is not None:
                buff.selection_state.enter_shift_mode()
            original_position = buff.cursor_position
            unshift_move(event)
            if buff.cursor_position == original_position:
                buff.exit_selection()

    @handle('s-left', filter=shift_selection_mode)
    @handle('s-right', filter=shift_selection_mode)
    @handle('s-up', filter=shift_selection_mode)
    @handle('s-down', filter=shift_selection_mode)
    @handle('s-home', filter=shift_selection_mode)
    @handle('s-end', filter=shift_selection_mode)
    @handle('c-s-left', filter=shift_selection_mode)
    @handle('c-s-right', filter=shift_selection_mode)
    @handle('c-s-home', filter=shift_selection_mode)
    @handle('c-s-end', filter=shift_selection_mode)
    def _extend_selection(event: E) -> None:
        """
        Extend the selection
        """
        unshift_move(event)
        buff = event.current_buffer
        if buff.selection_state is not None:
            if buff.cursor_position == buff.selection_state.original_cursor_position:
                buff.exit_selection()

    @handle(Keys.Any, filter=shift_selection_mode)
    def _replace_selection(event: E) -> None:
        """
        Replace selection by what is typed
        """
        event.current_buffer.cut_selection()
        get_by_name('self-insert').call(event)

    @handle('enter', filter=shift_selection_mode & is_multiline)
    def _newline(event: E) -> None:
        """
        A newline replaces the selection
        """
        event.current_buffer.cut_selection()
        event.current_buffer.newline(copy_margin=not in_paste_mode())

    @handle('backspace', filter=shift_selection_mode)
    def _delete(event: E) -> None:
        """
        Delete selection.
        """
        event.current_buffer.cut_selection()

    @handle('c-y', filter=shift_selection_mode)
    def _yank(event: E) -> None:
        """
        In shift selection mode, yanking (pasting) replace the selection.
        """
        buff = event.current_buffer
        if buff.selection_state:
            buff.cut_selection()
        get_by_name('yank').call(event)

    @handle('left', filter=shift_selection_mode)
    @handle('right', filter=shift_selection_mode)
    @handle('up', filter=shift_selection_mode)
    @handle('down', filter=shift_selection_mode)
    @handle('home', filter=shift_selection_mode)
    @handle('end', filter=shift_selection_mode)
    @handle('c-left', filter=shift_selection_mode)
    @handle('c-right', filter=shift_selection_mode)
    @handle('c-home', filter=shift_selection_mode)
    @handle('c-end', filter=shift_selection_mode)
    def _cancel(event: E) -> None:
        """
        Cancel selection.
        """
        event.current_buffer.exit_selection()
        key_press = event.key_sequence[0]
        event.key_processor.feed(key_press, first=True)
    return ConditionalKeyBindings(key_bindings, emacs_mode)