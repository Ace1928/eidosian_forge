import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@patch('colorama.winterm.win32')
def testSetConsoleOnStderr(self, mockWin32):
    mockAttr = Mock()
    mockAttr.wAttributes = 0
    mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
    term = WinTerm()
    term.windll = Mock()
    term.set_console(on_stderr=True)
    self.assertEqual(mockWin32.SetConsoleTextAttribute.call_args, ((mockWin32.STDERR, term.get_attrs()), {}))