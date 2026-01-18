from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def test_native_windows_ansi(self):
    with ExitStack() as stack:

        def p(a, b):
            stack.enter_context(patch(a, b, create=True))
        p('colorama.ansitowin32.os.name', 'nt')
        p('colorama.ansitowin32.winapi_test', lambda: True)
        p('colorama.win32.winapi_test', lambda: True)
        p('colorama.winterm.win32.windll', 'non-None')
        p('colorama.winterm.get_osfhandle', lambda _: 1234)
        p('colorama.winterm.win32.GetConsoleMode', lambda _: ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        SetConsoleMode = Mock()
        p('colorama.winterm.win32.SetConsoleMode', SetConsoleMode)
        stdout = Mock()
        stdout.closed = False
        stdout.isatty.return_value = True
        stdout.fileno.return_value = 1
        stream = AnsiToWin32(stdout)
        SetConsoleMode.assert_called_with(1234, ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        self.assertFalse(stream.strip)
        self.assertFalse(stream.convert)
        self.assertFalse(stream.should_wrap())
        p('colorama.winterm.win32.GetConsoleMode', lambda _: 0)
        SetConsoleMode = Mock()
        p('colorama.winterm.win32.SetConsoleMode', SetConsoleMode)
        stream = AnsiToWin32(stdout)
        SetConsoleMode.assert_called_with(1234, ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        self.assertTrue(stream.strip)
        self.assertTrue(stream.convert)
        self.assertTrue(stream.should_wrap())