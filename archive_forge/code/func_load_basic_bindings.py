from __future__ import unicode_literals
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import HasSelection, Condition, EmacsInsertMode, ViInsertMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.renderer import HeightIsUnknownError
from prompt_toolkit.utils import suspend_to_background_supported, is_windows
from .named_commands import get_by_name
from ..registry import Registry
def load_basic_bindings():
    registry = Registry()
    insert_mode = ViInsertMode() | EmacsInsertMode()
    handle = registry.add_binding
    has_selection = HasSelection()

    @handle(Keys.ControlA)
    @handle(Keys.ControlB)
    @handle(Keys.ControlC)
    @handle(Keys.ControlD)
    @handle(Keys.ControlE)
    @handle(Keys.ControlF)
    @handle(Keys.ControlG)
    @handle(Keys.ControlH)
    @handle(Keys.ControlI)
    @handle(Keys.ControlJ)
    @handle(Keys.ControlK)
    @handle(Keys.ControlL)
    @handle(Keys.ControlM)
    @handle(Keys.ControlN)
    @handle(Keys.ControlO)
    @handle(Keys.ControlP)
    @handle(Keys.ControlQ)
    @handle(Keys.ControlR)
    @handle(Keys.ControlS)
    @handle(Keys.ControlT)
    @handle(Keys.ControlU)
    @handle(Keys.ControlV)
    @handle(Keys.ControlW)
    @handle(Keys.ControlX)
    @handle(Keys.ControlY)
    @handle(Keys.ControlZ)
    @handle(Keys.F1)
    @handle(Keys.F2)
    @handle(Keys.F3)
    @handle(Keys.F4)
    @handle(Keys.F5)
    @handle(Keys.F6)
    @handle(Keys.F7)
    @handle(Keys.F8)
    @handle(Keys.F9)
    @handle(Keys.F10)
    @handle(Keys.F11)
    @handle(Keys.F12)
    @handle(Keys.F13)
    @handle(Keys.F14)
    @handle(Keys.F15)
    @handle(Keys.F16)
    @handle(Keys.F17)
    @handle(Keys.F18)
    @handle(Keys.F19)
    @handle(Keys.F20)
    @handle(Keys.ControlSpace)
    @handle(Keys.ControlBackslash)
    @handle(Keys.ControlSquareClose)
    @handle(Keys.ControlCircumflex)
    @handle(Keys.ControlUnderscore)
    @handle(Keys.Backspace)
    @handle(Keys.Up)
    @handle(Keys.Down)
    @handle(Keys.Right)
    @handle(Keys.Left)
    @handle(Keys.ShiftUp)
    @handle(Keys.ShiftDown)
    @handle(Keys.ShiftRight)
    @handle(Keys.ShiftLeft)
    @handle(Keys.Home)
    @handle(Keys.End)
    @handle(Keys.Delete)
    @handle(Keys.ShiftDelete)
    @handle(Keys.ControlDelete)
    @handle(Keys.PageUp)
    @handle(Keys.PageDown)
    @handle(Keys.BackTab)
    @handle(Keys.Tab)
    @handle(Keys.ControlLeft)
    @handle(Keys.ControlRight)
    @handle(Keys.ControlUp)
    @handle(Keys.ControlDown)
    @handle(Keys.Insert)
    @handle(Keys.Ignore)
    def _(event):
        """
        First, for any of these keys, Don't do anything by default. Also don't
        catch them in the 'Any' handler which will insert them as data.

        If people want to insert these characters as a literal, they can always
        do by doing a quoted insert. (ControlQ in emacs mode, ControlV in Vi
        mode.)
        """
        pass
    handle(Keys.Home)(get_by_name('beginning-of-line'))
    handle(Keys.End)(get_by_name('end-of-line'))
    handle(Keys.Left)(get_by_name('backward-char'))
    handle(Keys.Right)(get_by_name('forward-char'))
    handle(Keys.ControlUp)(get_by_name('previous-history'))
    handle(Keys.ControlDown)(get_by_name('next-history'))
    handle(Keys.ControlL)(get_by_name('clear-screen'))
    handle(Keys.ControlK, filter=insert_mode)(get_by_name('kill-line'))
    handle(Keys.ControlU, filter=insert_mode)(get_by_name('unix-line-discard'))
    handle(Keys.ControlH, filter=insert_mode, save_before=if_no_repeat)(get_by_name('backward-delete-char'))
    handle(Keys.Backspace, filter=insert_mode, save_before=if_no_repeat)(get_by_name('backward-delete-char'))
    handle(Keys.Delete, filter=insert_mode, save_before=if_no_repeat)(get_by_name('delete-char'))
    handle(Keys.ShiftDelete, filter=insert_mode, save_before=if_no_repeat)(get_by_name('delete-char'))
    handle(Keys.Any, filter=insert_mode, save_before=if_no_repeat)(get_by_name('self-insert'))
    handle(Keys.ControlT, filter=insert_mode)(get_by_name('transpose-chars'))
    handle(Keys.ControlW, filter=insert_mode)(get_by_name('unix-word-rubout'))
    handle(Keys.ControlI, filter=insert_mode)(get_by_name('menu-complete'))
    handle(Keys.BackTab, filter=insert_mode)(get_by_name('menu-complete-backward'))
    handle(Keys.PageUp, filter=~has_selection)(get_by_name('previous-history'))
    handle(Keys.PageDown, filter=~has_selection)(get_by_name('next-history'))
    text_before_cursor = Condition(lambda cli: cli.current_buffer.text)
    handle(Keys.ControlD, filter=text_before_cursor & insert_mode)(get_by_name('delete-char'))
    is_multiline = Condition(lambda cli: cli.current_buffer.is_multiline())
    is_returnable = Condition(lambda cli: cli.current_buffer.accept_action.is_returnable)

    @handle(Keys.ControlJ, filter=is_multiline & insert_mode)
    def _(event):
        """ Newline (in case of multiline input. """
        event.current_buffer.newline(copy_margin=not event.cli.in_paste_mode)

    @handle(Keys.ControlJ, filter=~is_multiline & is_returnable)
    def _(event):
        """ Enter, accept input. """
        buff = event.current_buffer
        buff.accept_action.validate_and_handle(event.cli, buff)

    @handle(Keys.Up)
    def _(event):
        event.current_buffer.auto_up(count=event.arg)

    @handle(Keys.Down)
    def _(event):
        event.current_buffer.auto_down(count=event.arg)

    @handle(Keys.Delete, filter=has_selection)
    def _(event):
        data = event.current_buffer.cut_selection()
        event.cli.clipboard.set_data(data)

    @handle(Keys.ControlZ)
    def _(event):
        """
        By default, control-Z should literally insert Ctrl-Z.
        (Ansi Ctrl-Z, code 26 in MSDOS means End-Of-File.
        In a Python REPL for instance, it's possible to type
        Control-Z followed by enter to quit.)

        When the system bindings are loaded and suspend-to-background is
        supported, that will override this binding.
        """
        event.current_buffer.insert_text(event.data)

    @handle(Keys.CPRResponse)
    def _(event):
        """
        Handle incoming Cursor-Position-Request response.
        """
        row, col = map(int, event.data[2:-1].split(';'))
        event.cli.renderer.report_absolute_cursor_row(row)

    @handle(Keys.BracketedPaste)
    def _(event):
        """ Pasting from clipboard. """
        data = event.data
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        event.current_buffer.insert_text(data)

    @handle(Keys.Any, filter=Condition(lambda cli: cli.quoted_insert), eager=True)
    def _(event):
        """
        Handle quoted insert.
        """
        event.current_buffer.insert_text(event.data, overwrite=False)
        event.cli.quoted_insert = False
    return registry