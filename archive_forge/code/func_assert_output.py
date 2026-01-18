import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def assert_output(self, expected):
    """Asserts that the current output string contains the expected."""
    output = self._captured.after
    decoded = output.decode('utf-8')
    assert expected in decoded, f'decoded={decoded}\nexpected={expected})'