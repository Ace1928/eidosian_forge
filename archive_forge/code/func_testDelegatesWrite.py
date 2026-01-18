from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testDelegatesWrite(self):
    mockStream = Mock()
    mockConverter = Mock()
    wrapper = StreamWrapper(mockStream, mockConverter)
    wrapper.write('hello')
    self.assertTrue(mockConverter.write.call_args, (('hello',), {}))