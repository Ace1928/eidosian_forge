from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testCallWin32UsesLookup(self):
    listener = Mock()
    stream = AnsiToWin32(listener)
    stream.win32_calls = {1: (lambda *_, **__: listener(11),), 2: (lambda *_, **__: listener(22),), 3: (lambda *_, **__: listener(33),)}
    stream.call_win32('m', (3, 1, 99, 2))
    self.assertEqual([a[0][0] for a in listener.call_args_list], [33, 11, 22])