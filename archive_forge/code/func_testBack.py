import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@skipUnless(sys.platform.startswith('win'), 'requires Windows')
def testBack(self):
    term = WinTerm()
    term.set_console = Mock()
    term._back = 0
    term.back(5)
    self.assertEqual(term._back, 5)
    self.assertEqual(term.set_console.called, True)