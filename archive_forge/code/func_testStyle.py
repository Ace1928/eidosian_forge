import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@skipUnless(sys.platform.startswith('win'), 'requires Windows')
def testStyle(self):
    term = WinTerm()
    term.set_console = Mock()
    term._style = 0
    term.style(22)
    self.assertEqual(term._style, 22)
    self.assertEqual(term.set_console.called, True)