from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testIsAProxy(self):
    mockStream = Mock()
    wrapper = StreamWrapper(mockStream, None)
    self.assertTrue(wrapper.random_attr is mockStream.random_attr)