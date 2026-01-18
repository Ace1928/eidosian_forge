import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@patch('colorama.winterm.win32')
def testResetAll(self, mockWin32):
    mockAttr = Mock()
    mockAttr.wAttributes = 1 + 2 * 16 + 8
    mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
    term = WinTerm()
    term.set_console = Mock()
    term._fore = -1
    term._back = -1
    term._style = -1
    term.reset_all()
    self.assertEqual(term._fore, 1)
    self.assertEqual(term._back, 2)
    self.assertEqual(term._style, 8)
    self.assertEqual(term.set_console.called, True)