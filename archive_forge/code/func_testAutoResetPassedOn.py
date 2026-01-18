import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
@patch('colorama.win32.SetConsoleTextAttribute')
@patch('colorama.initialise.AnsiToWin32')
def testAutoResetPassedOn(self, mockATW32, _):
    with osname('nt'):
        init(autoreset=True)
        self.assertEqual(len(mockATW32.call_args_list), 2)
        self.assertEqual(mockATW32.call_args_list[1][1]['autoreset'], True)
        self.assertEqual(mockATW32.call_args_list[0][1]['autoreset'], True)