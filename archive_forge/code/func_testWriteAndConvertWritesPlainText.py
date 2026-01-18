from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testWriteAndConvertWritesPlainText(self):
    stream = AnsiToWin32(Mock())
    stream.write_and_convert('abc')
    self.assertEqual(stream.wrapped.write.call_args, (('abc',), {}))