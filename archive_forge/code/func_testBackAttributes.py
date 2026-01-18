import sys
from unittest import TestCase, main
from ..ansi import Back, Fore, Style
from ..ansitowin32 import AnsiToWin32
def testBackAttributes(self):
    self.assertEqual(Back.BLACK, '\x1b[40m')
    self.assertEqual(Back.RED, '\x1b[41m')
    self.assertEqual(Back.GREEN, '\x1b[42m')
    self.assertEqual(Back.YELLOW, '\x1b[43m')
    self.assertEqual(Back.BLUE, '\x1b[44m')
    self.assertEqual(Back.MAGENTA, '\x1b[45m')
    self.assertEqual(Back.CYAN, '\x1b[46m')
    self.assertEqual(Back.WHITE, '\x1b[47m')
    self.assertEqual(Back.RESET, '\x1b[49m')
    self.assertEqual(Back.LIGHTBLACK_EX, '\x1b[100m')
    self.assertEqual(Back.LIGHTRED_EX, '\x1b[101m')
    self.assertEqual(Back.LIGHTGREEN_EX, '\x1b[102m')
    self.assertEqual(Back.LIGHTYELLOW_EX, '\x1b[103m')
    self.assertEqual(Back.LIGHTBLUE_EX, '\x1b[104m')
    self.assertEqual(Back.LIGHTMAGENTA_EX, '\x1b[105m')
    self.assertEqual(Back.LIGHTCYAN_EX, '\x1b[106m')
    self.assertEqual(Back.LIGHTWHITE_EX, '\x1b[107m')