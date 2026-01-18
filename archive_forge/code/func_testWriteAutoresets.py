from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testWriteAutoresets(self):
    self.assert_autoresets(convert=True)
    self.assert_autoresets(convert=False)
    self.assert_autoresets(convert=True, autoreset=False)
    self.assert_autoresets(convert=False, autoreset=False)