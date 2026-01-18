import sys
from unittest import TestCase, main
from ..ansi import Back, Fore, Style
from ..ansitowin32 import AnsiToWin32
def testStyleAttributes(self):
    self.assertEqual(Style.DIM, '\x1b[2m')
    self.assertEqual(Style.NORMAL, '\x1b[22m')
    self.assertEqual(Style.BRIGHT, '\x1b[1m')