from __future__ import absolute_import
import time
import msvcrt  # pylint: disable=import-error
import contextlib
from jinxed import win32  # pylint: disable=import-error
from .terminal import WINSZ
from .terminal import Terminal as _Terminal
@contextlib.contextmanager
def cbreak(self):
    """
        Allow each keystroke to be read immediately after it is pressed.

        This is a context manager for ``jinxed.w32.setcbreak()``.

        .. note:: You must explicitly print any user input you would like
            displayed.  If you provide any kind of editing, you must handle
            backspace and other line-editing control functions in this mode
            as well!

        **Normally**, characters received from the keyboard cannot be read
        by Python until the *Return* key is pressed. Also known as *cooked* or
        *canonical input* mode, it allows the tty driver to provide
        line-editing before shuttling the input to your program and is the
        (implicit) default terminal mode set by most unix shells before
        executing programs.
        """
    if self._keyboard_fd is not None:
        filehandle = msvcrt.get_osfhandle(self._keyboard_fd)
        save_mode = win32.get_console_mode(filehandle)
        save_line_buffered = self._line_buffered
        win32.setcbreak(filehandle)
        try:
            self._line_buffered = False
            yield
        finally:
            win32.set_console_mode(filehandle, save_mode)
            self._line_buffered = save_line_buffered
    else:
        yield