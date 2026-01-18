from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testWriteStripsAnsi(self):
    mockStdout = Mock()
    stream = AnsiToWin32(mockStdout)
    stream.wrapped = Mock()
    stream.write_and_convert = Mock()
    stream.strip = True
    stream.write('abc')
    self.assertFalse(stream.wrapped.write.called)
    self.assertEqual(stream.write_and_convert.call_args, (('abc',), {}))