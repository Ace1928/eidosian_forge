import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
@patch('colorama.initialise.AnsiToWin32')
def testAutoResetChangeable(self, mockATW32):
    with osname('nt'):
        init()
        init(autoreset=True)
        self.assertEqual(len(mockATW32.call_args_list), 4)
        self.assertEqual(mockATW32.call_args_list[2][1]['autoreset'], True)
        self.assertEqual(mockATW32.call_args_list[3][1]['autoreset'], True)
        init()
        self.assertEqual(len(mockATW32.call_args_list), 6)
        self.assertEqual(mockATW32.call_args_list[4][1]['autoreset'], False)
        self.assertEqual(mockATW32.call_args_list[5][1]['autoreset'], False)