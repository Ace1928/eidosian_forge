import sys
from unittest import TestCase, main, skipUnless
from ..winterm import WinColor, WinStyle, WinTerm
@skipUnless(sys.platform.startswith('win'), 'requires Windows')
def testFore(self):
    term = WinTerm()
    term.set_console = Mock()
    term._fore = 0
    term.fore(5)
    self.assertEqual(term._fore, 5)
    self.assertEqual(term.set_console.called, True)