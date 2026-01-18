from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testWriteAndConvertSkipsEmptySnippets(self):
    stream = AnsiToWin32(Mock())
    stream.call_win32 = Mock()
    stream.write_and_convert('\x1b[40m\x1b[41m')
    self.assertFalse(stream.wrapped.write.called)