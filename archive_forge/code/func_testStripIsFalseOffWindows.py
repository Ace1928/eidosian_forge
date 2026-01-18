from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testStripIsFalseOffWindows(self):
    with osname('posix'):
        mockStdout = Mock(closed=False)
        stream = AnsiToWin32(mockStdout)
        self.assertFalse(stream.strip)